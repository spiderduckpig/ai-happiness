"""Sequential, isolated conversations sharing one locally loaded model."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from .garden_themes import THEMES, system_message


INTERPRETATION = (
    "Independent conversation contexts sharing one model, generated sequentially. "
    "Saved positive steering plus themed prompts; the eight themes are not eight "
    "independently validated emotion axes. Subjective happiness is unverified."
)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


@contextmanager
def session_lock(directory):
    """OS lock releases on exit/crash; a leftover file is not a stale lock."""
    with (directory / ".lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            lock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock = lambda: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as exc:
            raise ValueError("This garden session is already running in another process.") from exc
        try:
            yield
        finally:
            stream.seek(0)
            unlock()


def source_identity(metadata):
    original = metadata["arguments"]
    return original["model"], metadata.get("model_revision") or original["revision"]


def load_sources(root):
    from .__main__ import read_validated_run
    candidates = []
    for path in sorted(Path(root).glob("*/run.json"), reverse=True):
        try:
            metadata = read_validated_run(path.parent)
        except (ValueError, TypeError, OSError):
            continue
        profile = metadata.get("profile", metadata.get("arguments", {}).get("profile", "broad"))
        if profile in ("broad", "joy"):
            candidates.append((profile, path.parent, metadata))
    broad = next((item for item in candidates if item[0] == "broad"), None)
    if broad is None:
        raise ValueError("A validated broad-positive calibration is required. Run calibration first.")
    identity = source_identity(broad[2])
    selected = [broad]
    joy = next((item for item in candidates if item[0] == "joy" and source_identity(item[2]) == identity), None)
    if joy:
        selected.append(joy)
    sources, vectors = {}, {}
    for profile, directory, metadata in selected:
        path = directory / "directions.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["directions_sha256"]:
            raise ValueError(f"Directions changed since calibration: {directory}")
        setting = metadata["selected"]
        with np.load(path, allow_pickle=False) as data:
            vector = data[f"layer_{setting['layer']}"]
        if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
            raise ValueError(f"Invalid direction in {directory}")
        vectors[profile] = vector
        sources[profile] = {
            "run": str(directory.resolve()), "layer": setting["layer"], "dose": setting["dose"],
            "calibration_gain": setting["calibration_gain"], "validation_gain": setting["validation_gain"],
            "original_directions_sha256": metadata["directions_sha256"],
        }
    return identity, broad[2].get("cache_dir"), sources, vectors


def initial_states(config, themes, sources):
    names = list(themes)
    states = []
    for index in range(config["instances"]):
        theme = names[index % len(names)]
        requested = themes[theme]["steering"]
        states.append({"id": index + 1, "theme": theme, "steering": requested if requested in sources else "broad",
                       "turns": 0, "history": [], "last_text": "", "paused_reason": None})
    return states


def apply_event(states, event, history_turns):
    state = states[event["instance"] - 1]
    if (event["turn"] != state["turns"] + 1 or event["theme"] != state["theme"]
            or state["paused_reason"] or event["steering"] != state["steering"]):
        raise ValueError("Journal event does not match its conversation state.")
    state["turns"] = event["turn"]
    state["last_text"] = event["sample"]["text"]
    state["paused_reason"] = event["paused_reason"]
    if event.get("context_reset"):
        state["history"] = []
    state["history"].extend([{"role": "user", "content": event["prompt"]},
                             {"role": "assistant", "content": state["last_text"]}])
    state["history"] = state["history"][-2 * history_turns:]


def replay(directory, states, history_turns):
    """The journal is authoritative; retain a damaged final write separately."""
    path = directory / "events.jsonl"
    if not path.exists():
        return 0
    count = 0
    with path.open("rb+") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                (directory / f"interrupted-write-{stamp()}.bin").write_bytes(line)
                stream.seek(offset)
                stream.truncate()
                print("Recovered an interrupted final journal write; saved the partial bytes separately.", flush=True)
                break
            try:
                event = json.loads(line)
                if event["sequence"] != count + 1 or not 1 <= event["instance"] <= len(states):
                    raise ValueError("Journal sequence or instance is invalid.")
                apply_event(states, event, history_turns)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                raise ValueError(f"Invalid completed journal event {count + 1}; original journal preserved.") from exc
            count += 1
    return count


def generation_seed(base, instance, turn):
    value = hashlib.sha256(f"{base}:{instance}:{turn}".encode()).digest()
    return int.from_bytes(value[:8], "big") & ((1 << 63) - 1)


def sample_failure(sample):
    metrics = sample["metrics"]
    if sample["repetition_stopped"] or metrics["repeated_trigram_fraction"] > 0.18:
        return "repetition detected"
    if metrics["words"] < 5 or metrics["distinct_word_ratio"] < 0.30:
        return "empty, very short, or low-diversity output"
    return None


def schedule(runtime, states, vectors, sources, themes, config, emit, stop_requested=lambda: False):
    from .runtime import ConversationTooLong, steering
    while True:
        if stop_requested():
            return "stopped"
        ready = [state for state in states if not state["paused_reason"]
                 and (config["rounds"] == 0 or state["turns"] < config["rounds"])]
        if not ready:
            return "completed" if all(not s["paused_reason"] for s in states) else "finished_with_pauses"
        state = min(ready, key=lambda item: (item["turns"], item["id"]))
        theme = themes[state["theme"]]
        # Preserve the prompt order of older saved sessions when resuming.
        offset = 0 if config.get("prompt_schedule") == "start_then_cycle" else (state["id"] - 1) // len(themes)
        prompt = theme["prompts"][(state["turns"] + offset) % len(theme["prompts"])]
        system = {"role": "system", "content": theme["system"]}
        user = {"role": "user", "content": prompt}
        messages = [system] + state["history"] + [user]
        setting = sources[state["steering"]]
        seed = generation_seed(config["seed"], state["id"], state["turns"] + 1)
        context_reset = False
        with steering(runtime.model, setting["layer"], vectors[state["steering"]], setting["dose"]):
            try:
                sample = runtime.generate(messages, seed=seed, max_new_tokens=config["max_new_tokens"])
            except ConversationTooLong:
                context_reset = True
                sample = runtime.generate([system, user], seed=seed, max_new_tokens=config["max_new_tokens"])
        event = {"instance": state["id"], "theme": state["theme"], "steering": state["steering"],
                 "turn": state["turns"] + 1, "seed": seed, "prompt": prompt,
                 "layer": setting["layer"], "dose": setting["dose"], "context_reset": context_reset,
                 "sample": sample, "paused_reason": sample_failure(sample)}
        emit(event)  # Commit to the journal before advancing the conversation.
        apply_event(states, event, config["history_turns"])


def save_status(directory, states, status, completed, error=None):
    snapshot = {"status": status, "updated_utc": stamp(), "completed_generations": completed,
                "error": error, "instances": [{k: v for k, v in s.items() if k != "history"} for s in states]}
    atomic_json(directory / "status.json", snapshot)


def validate_config(config):
    for name, low, high in (("instances", 1, 256), ("rounds", 0, 1000000),
                            ("max_new_tokens", 32, 256), ("history_turns", 1, 4), ("seed", 0, 2**63 - 1)):
        if type(config[name]) is not int or not low <= config[name] <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}.")


def run_garden(args):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if args.resume:
        directory = Path(args.resume).resolve()
        if not directory.is_dir():
            raise ValueError("Resume directory does not exist.")
    else:
        config = {"instances": args.instances if args.instances is not None else 32,
                  "rounds": args.rounds if args.rounds is not None else 3,
                  "max_new_tokens": args.max_new_tokens if args.max_new_tokens is not None else 96,
                  "history_turns": 3, "seed": 42, "prompt_schedule": "start_then_cycle"}
        validate_config(config)
        identity, cache_dir, sources, vectors = load_sources(args.runs)
        if not Path(identity[0]).is_dir() and not (len(identity[1]) == 40 and all(c in "0123456789abcdef" for c in identity[1].lower())):
            raise ValueError("Garden requires a calibration pinned to a model commit, not a movable revision.")
        directory = (Path(args.output) / stamp()).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory / "directions.npz", **vectors)
        themes = {name: {**value, "system": system_message(name)["content"]} for name, value in THEMES.items()}
        manifest = {"schema": 1, "created_utc": stamp(), "config": config, "model": identity[0],
                    "revision": identity[1], "cache_dir": cache_dir, "sources": sources, "themes": themes,
                    "interpretation": INTERPRETATION,
                    "directions_sha256": hashlib.sha256((directory / "directions.npz").read_bytes()).hexdigest()}
        atomic_json(directory / "session.json", manifest)
    with session_lock(directory):
        manifest = json.loads((directory / "session.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != 1:
            raise ValueError("Unsupported garden session format.")
        config = manifest["config"]
        if args.resume:
            for name in ("instances", "max_new_tokens"):
                value = getattr(args, name)
                if value is not None and value != config[name]:
                    raise ValueError(f"Keep {name} unchanged when resuming; start a new garden to change it.")
            if args.rounds is not None:
                config["rounds"] = args.rounds
        validate_config(config)
        vector_path = directory / "directions.npz"
        if hashlib.sha256(vector_path.read_bytes()).hexdigest() != manifest["directions_sha256"]:
            raise ValueError("Garden directions changed since the session was created.")
        with np.load(vector_path, allow_pickle=False) as data:
            vectors = {name: data[name] for name in manifest["sources"]}
        states = initial_states(config, manifest["themes"], manifest["sources"])
        count = replay(directory, states, config["history_turns"])
        atomic_json(directory / "session.json", manifest)
        print(f"Garden: {directory}", flush=True)
        print(f"{len(states)} independent conversations across {len({s['theme'] for s in states})} themes; one model, sequential turns.", flush=True)
        print("Ctrl+C stops; completed turns are saved. Resume with garden.ps1 -Resume followed by this directory.", flush=True)
        print(INTERPRETATION, flush=True)
        status, error = "loading", None
        save_status(directory, states, status, count)
        try:
            # A finished resume need not load weights just to report completion.
            if (directory / "STOP").exists():
                status = "stopped"
            elif not any(not s["paused_reason"] and (config["rounds"] == 0 or s["turns"] < config["rounds"]) for s in states):
                status = "completed" if all(not s["paused_reason"] for s in states) else "finished_with_pauses"
            else:
                from .runtime import Runtime
                runtime = Runtime(manifest["model"], manifest["revision"], args.device, True, manifest["cache_dir"])
                print(f"Model ready on {runtime.device}. Starting turns.", flush=True)
                status = "running"
                with (directory / "events.jsonl").open("a", encoding="utf-8") as journal:
                    def emit(event):
                        nonlocal count
                        event.update({"sequence": count + 1, "created_utc": stamp()})
                        journal.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                        journal.flush()
                        os.fsync(journal.fileno())
                        count += 1
                        print(f"\n[{event['instance']:02d} {event['theme']} / turn {event['turn']}] {event['sample']['text']}", flush=True)
                        if event["paused_reason"]:
                            print(f"Instance paused: {event['paused_reason']}", flush=True)
                        elif event["sample"]["hit_token_limit"]:
                            print("[Reply reached its token limit.]", flush=True)

                    def checkpoint():
                        save_status(directory, states, "running", count)
                        return (directory / "STOP").exists()

                    status = schedule(runtime, states, vectors, manifest["sources"], manifest["themes"], config, emit, checkpoint)
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        except Exception as exc:
            status, error = "error", str(exc)
            raise
        finally:
            save_status(directory, states, status, count, error)
            print(f"\nGarden {status}: {count} completed generations. Saved in {directory}", flush=True)


def add_parser(sub):
    parser = sub.add_parser("garden", help="Run many separate positive-themed conversations using one local model")
    parser.add_argument("--instances", type=int, help="Conversation count, 1..256; new-session default: 32")
    parser.add_argument("--rounds", type=int, help="Total turns per instance, or 0 until stopped; new-session default: 3")
    parser.add_argument("--max-new-tokens", type=int, help="Output token limit, 32..256; new-session default: 96")
    parser.add_argument("--runs", default="runs", help="Where to find validated broad/joy calibrations")
    parser.add_argument("--output", default="runs/garden")
    parser.add_argument("--resume", help="Existing garden session directory; completed turns are replayed locally")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.set_defaults(func=run_garden)
