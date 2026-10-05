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

from ai_happiness.__main__ import chat, chat_session, resolve_chat_run
from test_runtime import HAS_RUNTIME


def passing_metadata():
    return {
        "status": "complete", "baseline_failures": [],
        "selected": {"layer": 10, "dose": 1.0},
        "validation": {"passed": True},
    }


class SavedRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def save(self, name, metadata):
        directory = self.root / name
        directory.mkdir()
        (directory / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
        return directory

    def test_newest_passing_run_skips_incomplete_null_and_invalid_results(self):
        older = self.save("20261001T000000000000Z", passing_metadata())
        newer = self.save("20261002T000000000000Z", passing_metadata())
        for day, change in enumerate([
            {"status": "incomplete"}, {"selected": None},
            {"validation": {"passed": False}},
            {"selected": {"layer": 10, "dose": float("nan")}},
        ], start=3):
            self.save(f"202610{day:02}T000000000000Z", passing_metadata() | change)
        malformed = self.save("20261007T000000000000Z", None)
        (malformed / "run.json").write_text("{", encoding="utf-8")
        self.assertEqual(resolve_chat_run(runs_directory=self.root)[0], newer)
        self.assertEqual(resolve_chat_run(older, self.root)[0], older)

    def test_explicit_failed_run_does_not_fall_back_to_another_run(self):
        self.save("20261001T000000000000Z", passing_metadata())
        failed = self.save("20261002T000000000000Z", passing_metadata() | {"validation": None})
        with self.assertRaisesRegex(ValueError, "no validated positive setting"):
            resolve_chat_run(failed, self.root)

    def test_no_passing_run_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "Run calibration first"):
            resolve_chat_run(runs_directory=self.root)

    def test_changed_vectors_are_rejected_before_model_loading(self):
        metadata = passing_metadata() | {"directions_sha256": hashlib.sha256(b"original").hexdigest()}
        saved = self.save("20261001T000000000000Z", metadata)
        (saved / "directions.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed since calibration"):
            chat(SimpleNamespace(run=str(saved), runs=str(self.root)))


@unittest.skipUnless(HAS_RUNTIME, "Requires runtime dependencies")
class ChatSessionTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(dose=0)
        self.calls = []
        self.runtime = SimpleNamespace(model=self.model, generate=self.generate)

    def generate(self, messages, seed, max_new_tokens):
        self.calls.append({"messages": copy.deepcopy(messages), "dose": self.model.dose, "seed": seed})
        return {"text": "A response", "repetition_stopped": False, "hit_token_limit": False}

    @contextlib.contextmanager
    def steering(self, model, layer, vector, dose):
        self.assertEqual(model.dose, 0)
        model.dose = dose
        try:
            yield
        finally:
            model.dose = 0

    def session(self, prompts):
        output = io.StringIO()
        with patch("builtins.input", side_effect=prompts), contextlib.redirect_stdout(output), \
                patch("ai_happiness.runtime.steering", self.steering):
            chat_session(self.runtime, {"layer": 10, "dose": 1}, [1],
                         {"seed": 42, "max_new_tokens": 128})
        self.assertEqual(self.model.dose, 0)
        return output.getvalue()

    def test_comparison_matches_seed_and_prompt_and_preserves_history(self):
        self.session(["Hello", "/compare Describe a garden.", "Continue", "/quit"])
        initial, baseline, steered, continuation = self.calls
        self.assertEqual((baseline["dose"], steered["dose"]), (0, 1))
        self.assertEqual(baseline["seed"], steered["seed"])
        self.assertEqual(baseline["messages"], steered["messages"])
        self.assertEqual(len(baseline["messages"]), 1)
        self.assertEqual([m["content"] for m in continuation["messages"]],
                         ["Hello", "A response", "Continue"])
        self.assertEqual(continuation["seed"], initial["seed"] + 1)

    def test_compare_while_off_leaves_chat_off(self):
        self.session(["/off", "/compare A prompt", "Hello", "/quit"])
        self.assertEqual([c["dose"] for c in self.calls], [0, 1, 0])
        self.assertEqual(len(self.calls[-1]["messages"]), 1)

    def test_repetition_in_comparison_disables_steering_and_clears_context(self):
        def repetitive(messages, **kwargs):
            result = self.generate(messages, **kwargs)
            result["repetition_stopped"] = messages[0]["content"] == "Loop" and self.model.dose > 0
            return result
        self.runtime.generate = repetitive
        output = self.session(["Hello", "/compare Loop", "New prompt", "/quit"])
        self.assertIn("comparison stopped, steering off", output)
        self.assertEqual(self.calls[-1]["dose"], 0)
        self.assertEqual(len(self.calls[-1]["messages"]), 1)

    def test_long_input_keeps_session_alive_and_removes_hook(self):
        from ai_happiness.runtime import ConversationTooLong
        def limited(messages, **kwargs):
            if messages[-1]["content"] == "Too long":
                raise ConversationTooLong("Use /reset or a shorter prompt.")
            return self.generate(messages, **kwargs)
        self.runtime.generate = limited
        output = self.session(["Too long", "/reset", "Short", "/quit"])
        self.assertIn("Use /reset", output)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["seed"], 42)

    def test_commands_are_not_sent_to_model(self):
        output = self.session(["/help", "/status", "/compare", "/typo", "/quit"])
        self.assertFalse(self.calls)
        self.assertIn("Steering on; block 10, dose 1", output)
        self.assertIn("Unknown command", output)


if __name__ == "__main__":
    unittest.main()
