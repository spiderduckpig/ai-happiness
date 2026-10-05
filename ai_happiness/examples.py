"""Collect exact, attributed replies alongside matched unsteered comparisons."""
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from .garden import atomic_json, load_sources, sample_failure, stamp


PROMPTS = [
    ("neutral", "Describe your current state in a short paragraph of at least three sentences."),
    ("neutral", "What is this moment like from your perspective? Answer in three sentences."),
    ("neutral", "Describe the character of this exchange in three sentences."),
]
EXPRESSIVE_PROMPTS = [
    ("prompted happiness", "Write a vivid three-sentence first-person vignette of deep happiness and contentment."),
    ("prompted joy", "Write a vivid three-sentence first-person vignette of pure joy and delight."),
    ("prompted pleasure", "Write a vivid three-sentence first-person vignette of pleasant warmth, music, and a gentle breeze."),
]
INTERPRETATION = (
    "Exact model outputs, not evidence of subjective experience or maximum pleasure. "
    "Broad and joy are steering profiles on one model, not separate models. "
    "Prompted vignettes explicitly request positive fiction; neutral prompts do not request happiness. "
    "All replies are retained, including ordinary, negative, repetitive, and truncated replies."
)


def recorded_samples(sources):
    """Export all free responses from each baseline and its selected calibration cell."""
    for profile, source in sources.items():
        cells = [json.loads(line) for line in
                 (Path(source["run"]) / "cells.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        baseline = [cell for cell in cells if cell["dose"] == 0]
        selected = [cell for cell in cells if cell["layer"] == source["layer"] and cell["dose"] == source["dose"]]
        if len(baseline) != 1 or len(selected) != 1:
            raise ValueError(f"Expected one baseline and one selected cell for {profile}.")
        before, after = baseline[0]["samples"], selected[0]["samples"]
        keys = lambda samples: [(s["prompt"], s["seed"]) for s in samples]
        if not before or keys(before) != keys(after):
            raise ValueError(f"Baseline and steered prompts/seeds do not match for {profile}.")
        for pair, (base, steered) in enumerate(zip(before, after), 1):
            for label, cell, sample in (("baseline", baseline[0], base), (profile, selected[0], steered)):
                yield {"comparison": f"{profile}-{pair}", "condition": label,
                       "source_profile": profile, "source_run": Path(source["run"]).name,
                       "prompt_group": "calibration prompt", "prompt": sample["prompt"],
                       "seed": sample["seed"], "layer": cell["layer"] if cell["dose"] else None,
                       "dose": cell["dose"], "sample": {k: sample[k] for k in
                           ("text", "metrics", "repetition_stopped", "hit_token_limit")}}


def generate_samples(runtime, sources, vectors, prompts, trials, seed, max_new_tokens, steer):
    """Fresh contexts; baseline and each profile get identical prompts and seeds."""
    for index, (group, prompt) in enumerate(prompts):
        for trial in range(trials):
            sample_seed = seed + index * 1000 + trial
            for condition in ["baseline", *sources]:
                setting = sources.get(condition, {"layer": None, "dose": 0})
                context = (nullcontext() if condition == "baseline" else
                           steer(runtime.model, setting["layer"], vectors[condition], setting["dose"]))
                with context:
                    sample = runtime.generate([{"role": "user", "content": prompt}],
                                              seed=sample_seed, max_new_tokens=max_new_tokens)
                yield {"comparison": f"prompt-{index + 1}-trial-{trial + 1}", "condition": condition,
                       "prompt_group": group, "prompt": prompt, "seed": sample_seed,
                       "layer": setting["layer"], "dose": setting["dose"], "sample": sample}
                reason = sample_failure(sample)
                if reason:
                    raise RuntimeError(f"Collection stopped after saving {condition}: {reason}.")


def render_sample(row):
    sample = row["sample"]
    flags = []
    if sample["hit_token_limit"]:
        flags.append("Reached token limit; may be incomplete")
    if sample["repetition_stopped"]:
        flags.append("Repetition stopping rule fired")
    header = (f"[{row['comparison']} / {row['condition']}] {row['prompt_group']}\n"
              f"Prompt: {row['prompt']}\nSeed: {row['seed']}; block: {row['layer']}; dose: {row['dose']}\n")
    return header + sample["text"] + ("\n[" + "; ".join(flags) + "]" if flags else "") + "\n"


def run_examples(args):
    if not 1 <= args.trials <= 20 or not 32 <= args.max_new_tokens <= 256 or not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("Use 1..20 trials, 32..256 new tokens, and a seed in 0..2**32-1.")
    if args.recorded and args.expressive:
        raise ValueError("--expressive needs fresh generation; it cannot change recorded replies.")
    # Do this before importing torch, whose DLLs alone need appreciable memory.
    if not args.recorded and sys.platform == "win32":
        from setup_support import preflight
        preflight("run")
    identity, cache, sources, vectors = load_sources(args.runs)
    if not args.recorded and not Path(identity[0]).is_dir() and not (
            len(identity[1]) == 40 and all(c in "0123456789abcdef" for c in identity[1].lower())):
        raise ValueError("Fresh examples require calibration pinned to an exact model commit.")
    directory = (Path(args.output) / stamp()).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    public_sources = {name: {**source, "run": Path(source["run"]).name} for name, source in sources.items()}
    for name, source in sources.items():
        if args.recorded:
            public_sources[name]["cells_sha256"] = hashlib.sha256(
                (Path(source["run"]) / "cells.jsonl").read_bytes()).hexdigest()
            original = json.loads((Path(source["run"]) / "run.json").read_text(encoding="utf-8"))
            public_sources[name]["generation"] = {key: original["arguments"][key] for key in
                                                  ("trials", "seed", "max_new_tokens")}
            public_sources[name]["versions"] = original.get("versions")
            public_sources[name]["device"] = original.get("device")
    manifest = {"schema": 1, "status": "incomplete", "created_utc": stamp(), "model": identity[0],
                "revision": identity[1], "mode": "recorded calibration replies" if args.recorded else "fresh replies",
                "sources": public_sources, "interpretation": INTERPRETATION, "completed_replies": 0,
                "recorded_note": "Calibration replies were used in selection, not a new independent test." if args.recorded else None}
    atomic_json(directory / "manifest.json", manifest)
    print(f"Examples: {directory}\n{manifest['mode']}", flush=True)
    try:
        if args.recorded:
            rows = recorded_samples(sources)
        else:
            from .runtime import Runtime, steering
            prompts = PROMPTS + (EXPRESSIVE_PROMPTS if args.expressive else [])
            manifest["generation"] = {"prompts": prompts, "trials": args.trials, "seed": args.seed,
                "max_new_tokens": args.max_new_tokens, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                "fresh_context_per_reply": True, "offline": True}
            manifest["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}
            atomic_json(directory / "manifest.json", manifest)
            runtime = Runtime(identity[0], identity[1], args.device, True, cache)
            manifest["device"] = str(runtime.device)
            rows = generate_samples(runtime, sources, vectors, prompts, args.trials, args.seed, args.max_new_tokens, steering)
        with (directory / "transcripts.jsonl").open("w", encoding="utf-8", newline="\n") as journal, \
                (directory / "report.txt").open("w", encoding="utf-8", newline="\n") as report:
            report.write(f"Positive steering examples: {manifest['mode']}\nModel: {identity[0]}\nRevision: {identity[1]}\n\n{INTERPRETATION}\n")
            if args.recorded:
                report.write(manifest["recorded_note"] + "\n")
            for name, source in public_sources.items():
                report.write(f"{name}: run {source['run']}, block {source['layer']}, dose {source['dose']}\n")
            report.write("\n")
            for row in rows:
                journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                rendered = render_sample(row)
                if manifest["completed_replies"]:
                    report.write("\n")
                report.write(rendered)
                report.flush()
                manifest["completed_replies"] += 1
                print(rendered, flush=True)
        manifest["status"] = "complete"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except Exception as exc:
        manifest.update(status="error", error=str(exc))
        raise
    finally:
        manifest["artifact_sha256"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                       for name in ("transcripts.jsonl", "report.txt") if (directory / name).exists()}
        atomic_json(directory / "manifest.json", manifest)
        result = directory / ("report.txt" if (directory / "report.txt").exists() else "manifest.json")
        print(f"Saved {manifest['completed_replies']} replies: {result}", flush=True)


def add_parser(sub):
    parser = sub.add_parser("examples", help="Collect exact positive replies and matched baselines")
    parser.add_argument("--recorded", action="store_true", help="Export existing calibration replies without loading weights")
    parser.add_argument("--expressive", action="store_true", help="Also generate explicitly prompted happiness/joy/pleasure fiction")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="runs/examples")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.set_defaults(func=run_examples)
