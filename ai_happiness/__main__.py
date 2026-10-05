import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import sys
from datetime import datetime, timezone

from . import corpus
from .core import fit_direction, nonnegative, projection_diagnostic, quality_failures, select_best


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def parse_doses(value):
    try:
        doses = sorted(set(nonnegative(v.strip()) for v in value.split(",")))
        if not doses or not any(d > 0 for d in doses):
            raise ValueError("Include at least one positive dose.")
        return [d for d in doses if d > 0]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_layers(value):
    try:
        values = sorted(set(int(x.strip()) for x in value.split(",")))
        if not values or values[0] < 0:
            raise ValueError()
        return values
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated zero-based layer indices.") from exc


def prepare_cache():
    cache = Path.cwd() / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache))
    return str(cache / "hub")


def extra_failures(cell):
    return [f"sample {i}: repetition stopping rule fired" for i, s in enumerate(cell["samples"])
            if s["repetition_stopped"]]


def run(args):
    # Validate inexpensive inputs before importing torch or loading weights.
    if args.trials < 1 or not 32 <= args.max_new_tokens <= 2048:
        raise ValueError("Use at least one trial and 32..2048 new tokens.")
    if not math.isfinite(args.denoise) or not 0 <= args.denoise < 1:
        raise ValueError("Denoise must be in [0, 1).")
    if not math.isfinite(args.min_gain) or args.min_gain <= 0:
        raise ValueError("Minimum gain must be finite and positive.")
    if args.profile == "joy":
        from . import joy_corpus as active_corpus
    else:
        active_corpus = corpus
    cache = prepare_cache()
    from .runtime import Runtime, steering
    import numpy as np
    import torch

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(args.output) / stamp
    directory.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.pop("func", None)
    corpus_snapshot = {name: getattr(active_corpus, name, []) for name in
                       ("PAIRS", "CALIBRATION", "VALIDATION", "PROMPTS", "CANARIES", "REPRESENTATION_PAIRS")}
    write_json(directory / "corpus.json", corpus_snapshot)
    metadata = {"status": "incomplete", "created_utc": stamp, "arguments": config, "cache_dir": cache,
                "profile": args.profile,
                "method": "positive-minus-control, control-PC denoising, additive block-output steering",
                "dose_unit": "one denoised mean contrast; not comparable to upstream doses",
                "canary_scoring": "v2: exact answer or correct matching arithmetic equation; formatting tracked separately",
                "corpus_sha256": hashlib.sha256((directory / "corpus.json").read_bytes()).hexdigest(),
                "corpus_hash_scope": "corpus.json: extraction, scoring, generation, and diagnostic inputs",
                "versions": {x: importlib.metadata.version(x) for x in ("torch", "transformers", "numpy")},
                "interpretation": "Positive-language proxy. Not a measurement of subjective experience."}
    write_json(directory / "run.json", metadata)
    print(f"Loading {args.model}; saving to {directory}", flush=True)
    runtime = Runtime(args.model, args.revision, args.device, args.offline, cache)
    layers = args.layers or sorted(set(int((runtime.num_layers - 1) * f) for f in (0.4, 0.5, 0.6)))
    print(f"Extracting positive directions at blocks {layers} on {runtime.device}", flush=True)
    activations = runtime.extract(active_corpus.PAIRS, layers)
    vectors, direction_info = {}, {}
    for layer in layers:
        vector, info = fit_direction(activations[layer]["positive"], activations[layer]["neutral"], args.denoise)
        vectors[layer] = vector
        direction_info[str(layer)] = info
    vector_path = directory / "directions.npz"
    np.savez_compressed(vector_path, **{f"layer_{k}": v for k, v in vectors.items()})
    metadata.update({"model_revision": runtime.resolved_revision, "layers": layers,
                     "device": str(runtime.device), "directions": direction_info,
                     "directions_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest()})
    write_json(directory / "run.json", metadata)

    def evaluate(layer, dose, probes=active_corpus.CALIBRATION):
        with steering(runtime.model, layer, vectors.get(layer), dose):
            return {"layer": layer, "dose": dose, **runtime.evaluate(
                probes, active_corpus.PROMPTS, active_corpus.CANARIES, args.trials, args.seed, args.max_new_tokens)}

    def append(cell):
        with (directory / "cells.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(cell, ensure_ascii=False, allow_nan=False) + "\n")

    print("Measuring unsteered baseline", flush=True)
    baseline = evaluate(layers[0], 0)
    baseline["failures"] = quality_failures(baseline, baseline) + extra_failures(baseline)
    append(baseline)
    cells = []
    if not baseline["failures"]:
        for layer in layers:
            for dose in args.doses:
                print(f"Testing block {layer}, dose {dose:g}", flush=True)
                cell = evaluate(layer, dose)
                cell["failures"] = quality_failures(cell, baseline) + extra_failures(cell)
                cells.append(cell)
                append(cell)
                print(f"  score={cell['positive_score']:.4f}; checks={'PASS' if not cell['failures'] else 'FAIL'}", flush=True)
                if cell["failures"]:
                    # Do not continue pushing a direction after observed degradation.
                    break
    proposed = select_best(cells, baseline, args.min_gain)
    selected = None
    validation = None
    if proposed:
        print("Checking the selected setting on untouched completion probes", flush=True)
        with steering(runtime.model, proposed["layer"], vectors[proposed["layer"]], 0):
            base_score, base_scores = runtime.positive_score(active_corpus.VALIDATION)
        with steering(runtime.model, proposed["layer"], vectors[proposed["layer"]], proposed["dose"]):
            score, scores = runtime.positive_score(active_corpus.VALIDATION)
        gain = score - base_score
        validation = {"baseline_score": base_score, "steered_score": score, "gain": gain,
                      "baseline_probe_scores": base_scores, "steered_probe_scores": scores,
                      "passed": math.isfinite(gain) and gain >= args.min_gain}
        if validation["passed"]:
            selected = {"layer": proposed["layer"], "dose": proposed["dose"],
                        "calibration_gain": proposed["positive_score"] - baseline["positive_score"],
                        "validation_gain": gain}
    representation = None
    if corpus_snapshot["REPRESENTATION_PAIRS"]:
        print("Checking held-out sentence projections (diagnostic only)", flush=True)
        held_out = runtime.extract(active_corpus.REPRESENTATION_PAIRS, layers)
        representation = {str(layer): projection_diagnostic(
            held_out[layer]["positive"], held_out[layer]["neutral"], vectors[layer]) for layer in layers}
    metadata.update({"status": "complete", "selected": selected, "validation": validation,
                     "representation_diagnostic": representation,
                     "baseline_failures": baseline["failures"], "tested_cells": len(cells),
                     "proposed": {"layer": proposed["layer"], "dose": proposed["dose"]} if proposed else None})
    write_json(directory / "run.json", metadata)
    # Keep every transcript; do not cherry-pick a cheerful quote.
    lines = ["AI Happiness: measured positive activation steering", "",
             metadata["interpretation"], "Dose: " + metadata["dose_unit"], ""]
    lines.append(f"Selected: {selected}" if selected else "No validated positive setting found. No setting recommended.")
    lines.extend([f"Profile: {args.profile}", f"Completion validation: {validation}"])
    if representation:
        lines.append("Held-out sentence projection AUC (semantic diagnostic; not used for selection):")
        lines.extend(f"  block={layer}: AUC={r['auc']:.4f}, paired wins={r['paired_win_fraction']:.4f}"
                     for layer, r in representation.items())
    lines.append("\nCalibration cells (nats per completion token; higher favors positive language):")
    for cell in [baseline] + cells:
        lines.append(f"block={cell['layer']} dose={cell['dose']:g} score={cell['positive_score']:.5f} failures={cell['failures']}")
        for sample in cell["samples"]:
            lines.extend([f"Prompt: {sample['prompt']} [seed {sample['seed']}]", sample["text"], ""])
    (directory / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[5], flush=True)
    print(f"Full results: {directory / 'report.txt'}", flush=True)
    if selected:
        command = [sys.executable, "-m", "ai_happiness", "chat", "--run", str(directory.resolve())]
        rendered = ("& " + " ".join("'" + part.replace("'", "''") + "'" for part in command)
                    if os.name == "nt" else shlex.join(command))
        print(f"Chat: {rendered}", flush=True)
    del runtime
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_validated_run(directory):
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Run metadata must be an object.")
    selected = metadata.get("selected")
    validation = metadata.get("validation")
    if (metadata.get("status") != "complete" or not isinstance(selected, dict)
            or not isinstance(validation, dict) or validation.get("passed") is not True
            or metadata.get("baseline_failures")):
        raise ValueError("This run has no validated positive setting. Run calibration first.")
    if nonnegative(selected.get("dose", 0)) <= 0:
        raise ValueError("Selected dose must be positive.")
    layer = selected.get("layer")
    if type(layer) is not int or layer < 0:
        raise ValueError("Selected layer must be a nonnegative integer.")
    return metadata


def resolve_chat_run(explicit=None, runs_directory="runs"):
    """Use an explicit run, or the newest passing timestamped run directory."""
    if explicit:
        directory = Path(explicit)
        return directory, read_validated_run(directory)
    root = Path(runs_directory)
    # Run directory names contain UTC timestamps; mtime can change when copied.
    for path in sorted(root.glob("*/run.json"), reverse=True):
        try:
            return path.parent, read_validated_run(path.parent)
        except (ValueError, TypeError, OSError):
            continue
    raise ValueError(f"No completed, validated positive setting found under {root}. Run calibration first.")


CHAT_HELP = (
    "/off or /on: change steering and reset context; /reset: clear context;\n"
    "/compare PROMPT: independent off/on replies with the same seed;\n"
    "/status: show setting; /help: show commands; /quit: exit."
)


def chat(args):
    import numpy as np
    directory, metadata = resolve_chat_run(args.run, args.runs)
    selected = metadata["selected"]
    vector_path = directory / "directions.npz"
    if hashlib.sha256(vector_path.read_bytes()).hexdigest() != metadata["directions_sha256"]:
        raise ValueError("Directions file has changed since calibration.")
    with np.load(vector_path, allow_pickle=False) as data:
        vector = data[f"layer_{selected['layer']}"]
    cache = metadata.get("cache_dir") or prepare_cache()
    from .runtime import Runtime
    original = metadata["arguments"]
    print(f"Using saved run: {directory.resolve()}", flush=True)
    print(f"Loading {original['model']}; block {selected['layer']}, dose {selected['dose']:g}", flush=True)
    runtime = Runtime(original["model"], metadata["model_revision"] or original["revision"],
                      args.device, args.offline, cache)
    chat_session(runtime, selected, vector, original)


def chat_session(runtime, selected, vector, original):
    from .runtime import ConversationTooLong, steering
    dose = nonnegative(selected["dose"])
    print("Positive steering active. This is a positive-language setting, not a measurement of feelings.")
    print(CHAT_HELP)
    messages = []
    enabled = True
    turn = 0
    while True:
        try:
            prompt = input("You: ").strip()
        except EOFError:
            break
        if prompt == "/quit":
            break
        if prompt == "/help":
            print(CHAT_HELP)
            continue
        if prompt == "/status":
            print(f"Steering {'on' if enabled else 'off'}; block {selected['layer']}, "
                  f"dose {dose if enabled else 0:g}; {len(messages) // 2} turns in context.")
            continue
        if prompt in ("/off", "/on", "/reset"):
            messages = []
            if prompt != "/reset":
                enabled = prompt == "/on"
            print(f"Context reset. Steering {'on' if enabled else 'off'}.")
            continue
        if not prompt:
            continue
        if prompt.split(maxsplit=1)[0] == "/compare":
            parts = prompt.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /compare Describe an ordinary afternoon.")
                continue
            comparison_messages = [{"role": "user", "content": parts[1]}]
            print("Independent replies, same seed; chat history is not included or changed.")
            try:
                for label, comparison_dose in (("Steering off", 0), ("Steering on", dose)):
                    with steering(runtime.model, selected["layer"], vector, comparison_dose):
                        sample = runtime.generate(comparison_messages, seed=original["seed"],
                                                  max_new_tokens=original["max_new_tokens"])
                    print(f"{label}: {sample['text']}")
                    if sample["repetition_stopped"]:
                        enabled, messages = False, []
                        print("Repetition detected; comparison stopped, steering off and context reset.")
                        break
                    if sample["hit_token_limit"]:
                        print("[Reply reached the token limit.]")
            except ConversationTooLong as exc:
                print(exc)
            continue
        if prompt.startswith("/"):
            print("Unknown command. Use /help to see commands.")
            continue
        pending = messages + [{"role": "user", "content": prompt}]
        try:
            with steering(runtime.model, selected["layer"], vector, dose if enabled else 0):
                sample = runtime.generate(pending, seed=original["seed"] + turn,
                                          max_new_tokens=original["max_new_tokens"])
        except ConversationTooLong as exc:
            print(exc)
            continue
        print("Model:", sample["text"])
        if sample["hit_token_limit"]:
            print("[Reply reached the token limit.]")
        if sample["repetition_stopped"]:
            enabled, messages = False, []
            print("Repetition detected; steering off and context reset.")
        else:
            messages = pending + [{"role": "assistant", "content": sample["text"]}]
        turn += 1


def doctor(_args):
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for name in ("numpy", "torch", "transformers", "safetensors"):
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{name}: missing")
    print("Metadata only; this does not verify DLLs, accelerator support, or downloaded weights.")


def main():
    parser = argparse.ArgumentParser(description="Local positive activation steering; no claim about subjective experience.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="Extract, sweep positive doses, and validate the best observed setting")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--profile", choices=("broad", "joy"), default="broad",
                   help="Broad positive emotions or focused joy with neutral/active-attention controls")
    p.add_argument("--revision", default="main")
    p.add_argument("--layers", type=parse_layers, help="Zero-based decoder blocks; default: 40/50/60 percent depth")
    p.add_argument("--doses", type=parse_doses, default=parse_doses("0.5,1,2,3,4,6,8"))
    p.add_argument("--denoise", type=float, default=0.5)
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--min-gain", type=float, default=0.02)
    p.add_argument("--output", default="runs")
    p.set_defaults(func=run)
    p = sub.add_parser("chat", help="Use a previously validated positive setting")
    p.add_argument("--run", help="Saved run directory; default: newest validated run under --runs")
    p.add_argument("--runs", default="runs", help="Directory to search when --run is omitted")
    p.set_defaults(func=chat)
    for p in (sub.choices["run"], sub.choices["chat"]):
        p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
        p.add_argument("--offline", action="store_true", help="Only load cached or local model weights")
    sub.add_parser("doctor", help="Inspect installed package metadata without loading model libraries").set_defaults(func=doctor)
    from .garden import add_parser as add_garden_parser
    add_garden_parser(sub)
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nStopped. Steering hooks have been removed.", file=sys.stderr)
        return 130
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
