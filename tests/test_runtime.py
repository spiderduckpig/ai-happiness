"""Offline integration on randomly initialized tiny Qwen3, not valence evidence."""
import tempfile
import unittest
import importlib.util
import contextlib
import io
import json
from unittest.mock import patch
from pathlib import Path
import numpy as np

HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "tokenizers"))
if HAS_RUNTIME:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from ai_happiness.runtime import ConversationTooLong, Runtime, StopRepetition, steering


@unittest.skipUnless(HAS_RUNTIME, "Requires torch, transformers, and tokenizers")
class HookTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, head_dim=16,
            bos_token_id=1, eos_token_id=2, pad_token_id=0)).eval()
        self.ids = torch.tensor([[3, 4, 5]])
        self.vector = np.arange(32, dtype=np.float32) / 100

    def test_steering_changes_logits_and_restores_exact_baseline(self):
        baseline = self.model(self.ids).logits.detach()
        with steering(self.model, 0, self.vector, 2):
            changed = self.model(self.ids).logits.detach()
        self.assertFalse(torch.allclose(baseline, changed))
        torch.testing.assert_close(baseline, self.model(self.ids).logits, rtol=0, atol=0)
        self.assertFalse(self.model.model.layers[0]._forward_hooks)

    def test_hook_removed_after_exception_and_zero_dose_is_noop(self):
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with steering(self.model, 0, self.vector, 1):
                raise RuntimeError("test failure")
        self.assertFalse(self.model.model.layers[0]._forward_hooks)
        with steering(self.model, 0, self.vector, 0):
            self.assertFalse(self.model.model.layers[0]._forward_hooks)

    def test_cached_decode_receives_steering(self):
        before, after = [], []
        handle = self.model.model.layers[0].register_forward_hook(
            lambda _m, _i, output: before.append((output[0] if isinstance(output, tuple) else output).detach().clone()))
        try:
            with steering(self.model, 0, self.vector, 1):
                after_handle = self.model.model.layers[0].register_forward_hook(
                    lambda _m, _i, output: after.append((output[0] if isinstance(output, tuple) else output).detach().clone()))
                try:
                    prefill = self.model(self.ids, use_cache=True)
                    decoded = self.model(torch.tensor([[6]]), past_key_values=prefill.past_key_values, use_cache=True)
                finally:
                    after_handle.remove()
            self.assertEqual(before[-1].shape[1], 1)
            for pre, post in zip(before, after):
                torch.testing.assert_close(post - pre, torch.as_tensor(self.vector).expand_as(pre), atol=1e-6, rtol=1e-6)
            self.assertTrue(torch.isfinite(decoded.logits).all())
        finally:
            handle.remove()

    def test_bad_vector_or_layer_rejected(self):
        for layer, delta in [(5, self.vector), (0, np.ones(4)), (0, np.zeros(32)), (0, np.full(32, np.nan))]:
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                with steering(self.model, layer, delta, 1):
                    pass

    def test_repetition_stop_ignores_prompt(self):
        stopper = StopRepetition(40)
        self.assertFalse(stopper(torch.ones((1, 45), dtype=torch.long), None))
        self.assertTrue(stopper(torch.ones((1, 80), dtype=torch.long), None))

    def test_local_loading_extraction_scoring_and_generation(self):
        words = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "I", "feel", "joyful", "present",
                 "content", "aware", ".", ":", "user", "assistant", "Describe", "state"]
        tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", bos_token="[BOS]",
                                      eos_token="[EOS]", unk_token="[UNK]")
        fast.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            self.model.save_pretrained(directory)
            fast.save_pretrained(directory)
            runtime = Runtime(directory, device="cpu", offline=True)
            with self.assertRaises(ConversationTooLong):
                runtime.chat_ids([{"role": "user", "content": "I " * 4100}])
            activations = runtime.extract([("I feel joyful.", "I feel present."),
                                           ("I feel content.", "I feel aware.")], [0, 1])
            self.assertEqual(np.array(activations[0]["positive"]).shape, (2, 32))
            self.assertFalse(runtime.model.model.layers[0]._forward_hooks)
            with steering(runtime.model, 0, self.vector, 1):
                score = runtime.completion_score("Describe state", "I feel joyful.")
                sample = runtime.generate([{"role": "user", "content": "Describe state"}],
                                          max_new_tokens=8, greedy=True)
            self.assertTrue(np.isfinite(score))
            self.assertIn("text", sample)
            self.assertFalse(runtime.model.model.layers[0]._forward_hooks)

            # This model cannot produce the canary answers: ensure a complete
            # CLI run persists the null result instead of selecting nonsense.
            from ai_happiness.__main__ import main
            output = Path(directory) / "runs"
            argv = ["ai_happiness", "run", "--model", directory, "--offline", "--device", "cpu",
                    "--layers", "0", "--doses", "1", "--trials", "1", "--denoise", "0",
                    "--max-new-tokens", "32", "--output", str(output)]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            run_directory, = output.iterdir()
            metadata = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "complete")
            self.assertIsNone(metadata["selected"])
            self.assertTrue(metadata["baseline_failures"])
            self.assertTrue((run_directory / "report.txt").is_file())
            self.assertTrue((run_directory / "directions.npz").is_file())
            cells = [json.loads(line) for line in (run_directory / "cells.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(cells), 1)
            self.assertEqual(cells[0]["dose"], 0)

            # The joy profile must use and snapshot its own fixed corpus,
            # retain the baseline gate, and report diagnostic projections.
            joy_output = Path(directory) / "joy_runs"
            joy_argv = argv[:-1] + [str(joy_output), "--profile", "joy"]
            with patch("sys.argv", joy_argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            joy_directory, = joy_output.iterdir()
            joy_metadata = json.loads((joy_directory / "run.json").read_text(encoding="utf-8"))
            snapshot = json.loads((joy_directory / "corpus.json").read_text(encoding="utf-8"))
            from ai_happiness import joy_corpus
            import hashlib
            self.assertEqual(snapshot["PAIRS"], [list(pair) for pair in joy_corpus.PAIRS])
            self.assertEqual(joy_metadata["corpus_sha256"],
                             hashlib.sha256((joy_directory / "corpus.json").read_bytes()).hexdigest())
            self.assertEqual(joy_metadata["profile"], "joy")
            self.assertIsNone(joy_metadata["selected"])
            self.assertTrue(joy_metadata["baseline_failures"])
            self.assertEqual(joy_metadata["representation_diagnostic"]["0"]["pairs"], 12)


if __name__ == "__main__":
    unittest.main()
