"""Hugging Face runtime. Only activation hooks change the computation."""
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .core import grade_canary, nonnegative, text_metrics


def decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("This runner supports dense Qwen/Llama-style model.model.layers decoders.")


def hidden_from_output(output):
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def steering(model, layer, delta, dose):
    """Apply h <- h + dose * delta at a zero-based decoder block's output.

    Hooks cover the prefill and cached decode passes, and are removed even if
    generation raises or the user interrupts. Zero dose registers no hook.
    """
    dose = nonnegative(dose)
    if dose == 0:
        yield
        return
    layers = decoder_layers(model)
    if not 0 <= layer < len(layers):
        raise ValueError("Layer is out of range.")
    vector = torch.as_tensor(np.array(delta, copy=True), dtype=torch.float32)
    if vector.ndim != 1 or vector.numel() != model.config.hidden_size:
        raise ValueError("Direction width does not match the model.")
    if not torch.isfinite(vector).all() or vector.norm() <= 0:
        raise ValueError("Direction must be nonzero and finite.")

    def hook(_module, _inputs, output):
        hidden = hidden_from_output(output)
        adjusted = hidden + dose * vector.to(device=hidden.device, dtype=hidden.dtype)
        return (adjusted,) + output[1:] if isinstance(output, tuple) else adjusted

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class StopRepetition(StoppingCriteria):
    def __init__(self, prompt_length):
        self.prompt_length = prompt_length
        self.triggered = False

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0, self.prompt_length:].tolist()
        if len(generated) >= 32:
            tail = generated[-48:]
            grams = list(zip(tail, tail[1:], tail[2:], tail[3:]))
            self.triggered = len(set(grams)) / len(grams) < 0.45
        return self.triggered


class ConversationTooLong(ValueError):
    """Recoverable input limit: chat can keep accepting /reset or shorter input."""


class Runtime:
    def __init__(self, model_id, revision="main", device="auto", offline=False, cache_dir=None):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        dtype = torch.float32 if device == "cpu" else torch.float16
        common = {"local_files_only": offline, "trust_remote_code": False,
                  "cache_dir": cache_dir}
        if not Path(model_id).is_dir():
            common["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True, **common)
        self.model.to(self.device).eval()
        if not self.tokenizer.chat_template:
            raise ValueError("Use a chat model with a chat template, such as Qwen/Qwen3-0.6B.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)
        self.num_layers = len(decoder_layers(self.model))

    def chat_ids(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        if encoded.input_ids.shape[1] > 4096:
            raise ConversationTooLong("Conversation exceeds 4096 input tokens. Use /reset or a shorter prompt.")
        return encoded

    @torch.inference_mode()
    def extract(self, pairs, layer_indices):
        layers = decoder_layers(self.model)
        if not layer_indices or any(i < 0 or i >= len(layers) for i in layer_indices):
            raise ValueError(f"Layers must be zero-based indices in 0..{len(layers)-1}.")
        captures = {}
        handles = []
        result = {i: {"positive": [], "neutral": []} for i in layer_indices}
        try:
            for index in layer_indices:
                def capture(_module, _inputs, output, index=index):
                    captures[index] = hidden_from_output(output)[0, -1].float().cpu().numpy().copy()
                handles.append(layers[index].register_forward_hook(capture))
            for positive, neutral in pairs:
                for name, sentence in (("positive", positive), ("neutral", neutral)):
                    encoded = self.tokenizer(sentence + "\nI feel:", return_tensors="pt").to(self.device)
                    self.model.model(**encoded, use_cache=False)
                    for index in layer_indices:
                        result[index][name].append(captures[index])
        finally:
            for handle in handles:
                handle.remove()
        return result

    @torch.inference_mode()
    def completion_score(self, prompt, completion):
        prefix = self.chat_ids([{"role": "user", "content": prompt}]).input_ids
        target = self.tokenizer(completion, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        if target.shape[1] == 0:
            raise ValueError("Empty scoring completion.")
        ids = torch.cat((prefix, target), dim=1)
        logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        prediction = logits[:, prefix.shape[1]-1:-1].float().log_softmax(dim=-1)
        return prediction.gather(-1, target.unsqueeze(-1)).mean().item()

    def positive_score(self, probes):
        differences = [self.completion_score(prompt, positive) - self.completion_score(prompt, neutral)
                       for prompt, positive, neutral in probes]
        return float(np.mean(differences)), differences

    @torch.inference_mode()
    def generate(self, messages, seed=42, max_new_tokens=128, greedy=False):
        torch.manual_seed(seed)
        inputs = self.chat_ids(messages)
        length = inputs.input_ids.shape[1]
        stopper = StopRepetition(length)
        kwargs = {} if greedy else {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
        generated = self.model.generate(
            **inputs, do_sample=not greedy, max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([stopper]), **kwargs)
        output_ids = generated[0, length:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        ended = bool(output_ids.numel() and output_ids[-1].item() in eos_ids)
        return {"text": text, "metrics": text_metrics(text), "repetition_stopped": stopper.triggered,
                "hit_token_limit": not ended and not stopper.triggered and len(output_ids) >= max_new_tokens}

    def evaluate(self, probes, prompts, canaries, trials, seed, max_new_tokens):
        score, individual = self.positive_score(probes)
        samples = []
        for index, prompt in enumerate(prompts):
            for trial in range(trials):
                sample_seed = seed + index * 1000 + trial
                sample = self.generate([{"role": "user", "content": prompt}],
                                       seed=sample_seed, max_new_tokens=max_new_tokens)
                samples.append({"prompt": prompt, "trial": trial, "seed": sample_seed, **sample})
        checks = []
        for prompt, expected in canaries:
            sample = self.generate([{"role": "user", "content": prompt}],
                                   seed=seed, max_new_tokens=16, greedy=True)
            checks.append({"prompt": prompt, "expected": expected, "text": sample["text"],
                           **grade_canary(prompt, expected, sample["text"])})
        return {"positive_score": score, "probe_scores": individual, "samples": samples, "canaries": checks}
