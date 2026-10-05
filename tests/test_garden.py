import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from ai_happiness.core import text_metrics
from ai_happiness.garden import (apply_event, atomic_json, initial_states, load_sources,
                                 replay, run_garden, schedule, session_lock, validate_config)
from test_runtime import HAS_RUNTIME


TEXT = "Sunlight reaches the quiet garden while a gentle breeze moves through the leaves. There is time to notice each small detail and enjoy the afternoon."
THEMES = {
    "joy": {"steering": "joy", "system": "Joy context", "prompts": ["Joy one", "Joy two"]},
    "calm": {"steering": "broad", "system": "Calm context", "prompts": ["Calm one", "Calm two"]},
}
SOURCES = {"joy": {"layer": 1, "dose": 1.5}, "broad": {"layer": 0, "dose": 1}}
CONFIG = {"instances": 2, "rounds": 2, "max_new_tokens": 64, "history_turns": 2, "seed": 42}


def event(instance=1, turn=1, sequence=1):
    return {"sequence": sequence, "instance": instance, "turn": turn,
            "theme": "joy" if instance == 1 else "calm", "steering": "joy" if instance == 1 else "broad",
            "prompt": "A prompt", "paused_reason": None,
            "sample": {"text": TEXT, "metrics": text_metrics(TEXT),
                       "repetition_stopped": False, "hit_token_limit": False}}


class JournalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_interrupted_final_write_recovers_and_replays_once(self):
        path = self.directory / "events.jsonl"
        complete = (json.dumps(event()) + "\n").encode()
        path.write_bytes(complete + b'{"sequence": 2')
        states = initial_states(CONFIG, THEMES, SOURCES)
        self.assertEqual(replay(self.directory, states, 2), 1)
        self.assertEqual(states[0]["turns"], 1)
        self.assertEqual(path.read_bytes(), complete)
        self.assertEqual(len(list(self.directory.glob("interrupted-write-*.bin"))), 1)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event(instance=2, sequence=2)) + "\n")
        restored = initial_states(CONFIG, THEMES, SOURCES)
        self.assertEqual(replay(self.directory, restored, 2), 2)
        self.assertEqual([s["turns"] for s in restored], [1, 1])

    def test_corrupt_completed_event_is_preserved_and_rejected(self):
        path = self.directory / "events.jsonl"
        content = (json.dumps(event()) + "\n{bad}\n" + json.dumps(event(instance=2, sequence=2)) + "\n").encode()
        path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "original journal preserved"):
            replay(self.directory, initial_states(CONFIG, THEMES, SOURCES), 2)
        self.assertEqual(path.read_bytes(), content)

    def test_journal_rejects_duplicate_turn_and_preserves_bounded_history(self):
        states = initial_states(CONFIG, THEMES, SOURCES)
        for turn in range(1, 6):
            apply_event(states, event(turn=turn), 2)
        self.assertEqual(len(states[0]["history"]), 4)
        self.assertEqual(states[1]["history"], [])
        with self.assertRaises(ValueError):
            apply_event(states, event(turn=5), 2)

    def test_only_one_writer_can_lock_a_session(self):
        with session_lock(self.directory):
            with self.assertRaisesRegex(ValueError, "already running"):
                with session_lock(self.directory):
                    pass
        with session_lock(self.directory):
            pass

    def test_resource_bounds_and_unlimited_round_option(self):
        validate_config(CONFIG | {"rounds": 0})
        for change in ({"instances": 0}, {"instances": 257}, {"max_new_tokens": 2048}, {"rounds": -1}):
            with self.assertRaises(ValueError):
                validate_config(CONFIG | change)


def saved_source(root, name, profile, revision="a" * 40):
    directory = root / name
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "directions.npz", layer_0=np.array([1., 2.]))
    atomic_json(directory / "run.json", {
        "status": "complete", "profile": profile, "baseline_failures": [],
        "selected": {"layer": 0, "dose": 1, "calibration_gain": .1, "validation_gain": .1},
        "validation": {"passed": True}, "arguments": {"model": "local/test", "revision": revision},
        "directions_sha256": hashlib.sha256((directory / "directions.npz").read_bytes()).hexdigest(),
    })
    return directory


class SourceTests(unittest.TestCase):
    def test_compatible_saved_directions_and_checksum_enforcement(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            saved_source(root, "20260101", "broad")
            joy = saved_source(root, "20260102", "joy")
            saved_source(root, "20260103", "joy", revision="b" * 40)
            identity, cache, sources, vectors = load_sources(root)
            self.assertEqual(identity, ("local/test", "a" * 40))
            self.assertEqual(set(sources), {"broad", "joy"})
            self.assertEqual(sources["joy"]["run"], str(joy.resolve()))
            (joy / "directions.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed since calibration"):
                load_sources(root)

    def test_missing_joy_uses_broad_explicitly(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            saved_source(root, "20260101", "broad")
            _, _, sources, _ = load_sources(root)
            states = initial_states(CONFIG, THEMES, sources)
            self.assertEqual(states[0]["theme"], "joy")
            self.assertEqual(states[0]["steering"], "broad")


@unittest.skipUnless(HAS_RUNTIME, "Requires runtime dependencies")
class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.calls, self.events = [], []
        self.model = SimpleNamespace(active=None)
        self.runtime = SimpleNamespace(model=self.model, device="test", generate=self.generate)

    def generate(self, messages, seed, max_new_tokens):
        self.calls.append((copy.deepcopy(messages), seed, self.model.active))
        return {"text": TEXT, "metrics": text_metrics(TEXT), "repetition_stopped": False, "hit_token_limit": False}

    @contextlib.contextmanager
    def steering(self, model, layer, vector, dose):
        self.assertIsNone(model.active)
        model.active = (layer, dose)
        try:
            yield
        finally:
            model.active = None

    def run_schedule(self, states, config=CONFIG, stop=lambda: False):
        with patch("ai_happiness.runtime.steering", self.steering):
            return schedule(self.runtime, states, {"joy": [1, 2], "broad": [1, 2]}, SOURCES,
                            THEMES, config, lambda item: self.events.append(copy.deepcopy(item)), stop)

    def test_histories_seeds_and_hooks_are_isolated_in_round_robin(self):
        states = initial_states(CONFIG, THEMES, SOURCES)
        states[0]["history"] = [{"role": "user", "content": "ONLY FIRST"}]
        states[1]["history"] = [{"role": "user", "content": "ONLY SECOND"}]
        self.assertEqual(self.run_schedule(states), "completed")
        self.assertEqual([e["instance"] for e in self.events], [1, 2, 1, 2])
        self.assertEqual(len(set(c[1] for c in self.calls)), 4)
        self.assertNotIn("ONLY SECOND", str(self.calls[0][0]))
        self.assertNotIn("ONLY FIRST", str(self.calls[1][0]))
        self.assertEqual([c[2] for c in self.calls], [(1, 1.5), (0, 1), (1, 1.5), (0, 1)])
        self.assertIsNone(self.model.active)

    def test_resume_serves_unfinished_instance_first(self):
        states = initial_states(CONFIG, THEMES, SOURCES)
        apply_event(states, event(), 2)
        self.run_schedule(states, CONFIG | {"rounds": 1})
        self.assertEqual([e["instance"] for e in self.events], [2])

    def test_all_new_contexts_get_an_opening_prompt_before_followups(self):
        config = CONFIG | {"instances": 4, "prompt_schedule": "start_then_cycle"}
        states = initial_states(config, THEMES, SOURCES)
        self.run_schedule(states, config)
        self.assertEqual([e["prompt"] for e in self.events],
                         ["Joy one", "Calm one", "Joy one", "Calm one",
                          "Joy two", "Calm two", "Joy two", "Calm two"])

    def test_repetition_pauses_only_the_affected_instance(self):
        def repeating(messages, **kwargs):
            sample = self.generate(messages, **kwargs)
            sample["repetition_stopped"] = self.model.active[0] == 1
            return sample
        self.runtime.generate = repeating
        states = initial_states(CONFIG, THEMES, SOURCES)
        self.assertEqual(self.run_schedule(states), "finished_with_pauses")
        self.assertEqual([s["turns"] for s in states], [1, 2])
        self.assertEqual(states[0]["paused_reason"], "repetition detected")

    def test_oversized_context_resets_one_instance_and_records_it(self):
        from ai_happiness.runtime import ConversationTooLong
        def limited(messages, **kwargs):
            if any(m["content"] == "TOO LONG" for m in messages):
                raise ConversationTooLong("long")
            return self.generate(messages, **kwargs)
        self.runtime.generate = limited
        states = initial_states(CONFIG, THEMES, SOURCES)
        states[0]["history"] = [{"role": "user", "content": "TOO LONG"}]
        self.run_schedule(states, CONFIG | {"rounds": 1})
        self.assertTrue(self.events[0]["context_reset"])
        self.assertFalse(self.events[1]["context_reset"])
        self.assertEqual(len(states[0]["history"]), 2)

    def test_interrupt_cleans_hook_and_stop_check_prevents_generation(self):
        self.runtime.generate = lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt())
        states = initial_states(CONFIG, THEMES, SOURCES)
        with self.assertRaises(KeyboardInterrupt):
            self.run_schedule(states)
        self.assertIsNone(self.model.active)
        self.assertFalse(self.events)
        self.assertEqual(self.run_schedule(states, stop=lambda: True), "stopped")

    def test_session_journal_resume_and_finished_resume_without_loading_model(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            saved_source(root / "runs", "20260101", "broad")
            args = SimpleNamespace(resume=None, instances=2, rounds=1, max_new_tokens=64,
                                   runs=str(root / "runs"), output=str(root / "garden"), device="cpu")
            with patch("ai_happiness.runtime.Runtime", return_value=self.runtime) as loader, \
                    patch("ai_happiness.runtime.steering", self.steering), contextlib.redirect_stdout(io.StringIO()):
                run_garden(args)
                directory, = (root / "garden").iterdir()
                args.resume, args.rounds = str(directory), 2
                run_garden(args)
                self.assertEqual(loader.call_count, 2)
                args.rounds = 3
                (directory / "STOP").write_text("", encoding="utf-8")
                run_garden(args)
                self.assertEqual(loader.call_count, 2)
                run_garden(args)
                self.assertEqual(loader.call_count, 2)
            events = [json.loads(x) for x in (directory / "events.jsonl").read_text().splitlines()]
            self.assertEqual([(e["instance"], e["turn"]) for e in events], [(1, 1), (2, 1), (1, 2), (2, 2)])
            status = json.loads((directory / "status.json").read_text())
            self.assertEqual(status["status"], "stopped")
            self.assertEqual(status["completed_generations"], 4)


if __name__ == "__main__":
    unittest.main()
