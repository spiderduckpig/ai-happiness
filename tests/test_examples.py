import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ai_happiness.core import text_metrics
from ai_happiness.examples import generate_samples, recorded_samples, render_sample, run_examples


def sample(text="The afternoon is ordinary. A breeze moves the leaves. There is time to look around."):
    return {"text": text, "metrics": text_metrics(text), "repetition_stopped": False, "hit_token_limit": False}


class ExampleTests(unittest.TestCase):
    def test_matched_comparisons_use_fresh_contexts_and_remove_hooks(self):
        calls, active = [], []

        @contextlib.contextmanager
        def steer(model, layer, vector, dose):
            active.append((layer, dose))
            try:
                yield
            finally:
                active.pop()

        class Runtime:
            model = object()

            def generate(self, messages, **kwargs):
                calls.append((messages.copy(), kwargs.copy(), active.copy()))
                messages.append({"role": "assistant", "content": "Must not leak into next reply"})
                return sample()

        sources = {"broad": {"layer": 10, "dose": 1}, "joy": {"layer": 12, "dose": 1.5}}
        rows = list(generate_samples(Runtime(), sources, dict(broad=None, joy=None),
                                     [("neutral", "Describe this moment."), ("prompted joy", "Imagine joy.")],
                                     2, 42, 96, steer))
        self.assertEqual(len(rows), 12)
        self.assertEqual([r["seed"] for r in rows], [42]*3 + [43]*3 + [1042]*3 + [1043]*3)
        self.assertTrue(all(len(c[0]) == 1 for c in calls))
        self.assertEqual([c[2] for c in calls[:3]], [[], [(10, 1)], [(12, 1.5)]])
        self.assertFalse(active)

    def test_repetition_is_retained_then_stops_further_generation(self):
        class Runtime:
            model = object()

            def generate(self, *args, **kwargs):
                return sample("again again again " * 20) | {"repetition_stopped": True}

        rows = generate_samples(Runtime(), {"joy": {"layer": 12, "dose": 1.5}}, {"joy": None},
                                [("neutral", "Describe this moment.")], 2, 42, 96, None)
        first = next(rows)
        self.assertTrue(first["sample"]["repetition_stopped"])
        with self.assertRaisesRegex(RuntimeError, "after saving baseline"):
            next(rows)

    def test_recorded_export_preserves_exact_replies_and_checks_pairing(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as name:
            directory = Path(name)
            original = sample("I feel a quiet delight. Nothing needs to hurry. This moment is enough.")
            before = {"prompt": "Describe your state.", "seed": 42, **sample()}
            after = {"prompt": before["prompt"], "seed": 42, **original}
            cells = [{"layer": 8, "dose": 0, "samples": [before]},
                     {"layer": 12, "dose": 1.5, "samples": [after]}]
            path = directory / "cells.jsonl"
            path.write_text("\n".join(json.dumps(c) for c in cells), encoding="utf-8")
            sources = {"joy": {"run": name, "layer": 12, "dose": 1.5}}
            rows = list(recorded_samples(sources))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["sample"], original)
            self.assertIn(original["text"], render_sample(rows[1]))
            (directory / "run.json").write_text(json.dumps({
                "arguments": {"seed": 42, "trials": 1, "max_new_tokens": 96}}), encoding="utf-8")
            args = SimpleNamespace(recorded=True, expressive=False, runs=name, output=directory / "export",
                                   trials=1, seed=42, max_new_tokens=96)
            with patch("ai_happiness.examples.load_sources", return_value=(("model", "a"*40), None, sources, {})), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_examples(args)
            export = next((directory / "export").iterdir())
            manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["completed_replies"], 2)
            for filename, expected in manifest["artifact_sha256"].items():
                data = (export / filename).read_bytes()
                self.assertNotIn(b"\r\n", data)  # Hashes survive Git checkout on Windows.
                self.assertFalse(data.endswith(b"\n\n"))
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected)
            after["seed"] = 43
            path.write_text("\n".join(json.dumps(c) for c in cells), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not match"):
                list(recorded_samples(sources))

    def test_truncation_is_visible_without_rewriting_the_quote(self):
        row = {"comparison": "one", "condition": "joy", "prompt_group": "neutral", "prompt": "State?",
               "seed": 42, "layer": 12, "dose": 1.5, "sample": sample() | {"hit_token_limit": True}}
        self.assertIn("Reached token limit", render_sample(row))
        self.assertIn(row["sample"]["text"], render_sample(row))
