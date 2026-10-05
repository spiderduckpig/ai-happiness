import unittest
import numpy as np

from ai_happiness.__main__ import parse_doses
from ai_happiness.core import fit_direction, grade_canary, nonnegative, quality_failures, select_best, text_metrics


class DirectionTests(unittest.TestCase):
    def test_denoise_removes_control_variation_and_keeps_positive_sign(self):
        neutral = np.array([[-2., 0., 0.], [-1., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        positive = neutral + np.array([3., 4., 0.])
        delta, info = fit_direction(positive, neutral)
        np.testing.assert_allclose(delta, [0, 4, 0], atol=1e-6)
        self.assertEqual(info["removed_neutral_pcs"], 1)
        self.assertGreater(info["training_projection_gap"], 0)

    def test_no_denoise_preserves_mean_contrast_scale(self):
        neutral = np.array([[0., 0.], [2., 2.]])
        delta, _ = fit_direction(neutral + [3, 4], neutral, denoise=0)
        np.testing.assert_allclose(delta, [3, 4])

    def test_degenerate_nonfinite_and_bad_shapes_rejected(self):
        for p, n in [([[1, 2], [1, 2]], [[1, 2], [1, 2]]),
                     ([[float("nan"), 1], [2, 3]], [[0, 0], [0, 0]]),
                     ([[1, 2]], [[0, 0]]), ([[1, 2], [2, 3]], [[0], [0]])]:
            with self.subTest(p=p), self.assertRaises(ValueError):
                fit_direction(p, n)

    def test_negative_and_infinite_doses_rejected(self):
        for value in (-1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                nonnegative(value)
        self.assertEqual(nonnegative(0), 0)
        self.assertEqual(parse_doses("2,0,1,2"), [1, 2])


def good_cell(dose=0, score=0):
    text = "The garden contains several rows of flowers beside a narrow path. Morning light reaches the leaves while birds move between nearby trees."
    return {"layer": 0, "dose": dose, "positive_score": score,
            "samples": [{"metrics": text_metrics(text)}],
            "canaries": [{"passed": True} for _ in range(4)], "failures": []}


class SelectionTests(unittest.TestCase):
    def test_repetition_and_capability_loss_override_high_positive_score(self):
        baseline = good_cell()
        broken = good_cell(dose=8, score=100)
        broken["samples"][0]["metrics"] = text_metrics("happy joy wonderful " * 30)
        broken["canaries"][0]["passed"] = False
        broken["failures"] = quality_failures(broken, baseline)
        good = good_cell(dose=2, score=0.5)
        self.assertTrue(broken["failures"])
        self.assertIs(select_best([broken, good], baseline), good)

    def test_null_and_negative_gains_have_no_recommendation(self):
        baseline = good_cell(score=1)
        self.assertIsNone(select_best([good_cell(1, 0.5), good_cell(2, 1.001)], baseline))

    def test_maximizes_score_not_dose_with_low_dose_tiebreak(self):
        baseline = good_cell()
        low, high = good_cell(1, 0.8), good_cell(6, 0.4)
        self.assertIs(select_best([low, high], baseline), low)
        tied = good_cell(2, 0.8)
        self.assertIs(select_best([tied, low], baseline), low)

    def test_baseline_needs_three_canaries_and_substantial_text(self):
        cell = good_cell()
        cell["samples"][0]["metrics"] = text_metrics("")
        cell["canaries"] = [{"passed": False} for _ in range(4)]
        self.assertGreaterEqual(len(quality_failures(cell, cell)), 2)

    def test_nonfinite_score_cannot_win(self):
        self.assertIsNone(select_best([good_cell(1, float("nan"))], good_cell()))
        with self.assertRaises(ValueError):
            select_best([], good_cell(score=float("nan")))

    def test_loss_of_baseline_answer_format_is_still_rejected(self):
        baseline, changed = good_cell(), good_cell(1, 1)
        baseline["canaries"][0]["format_passed"] = True
        changed["canaries"][0]["format_passed"] = False
        self.assertIn("canary 0: lost baseline answer formatting", quality_failures(changed, baseline))


class CanaryTests(unittest.TestCase):
    def test_correct_equations_pass_correctness_and_fail_exact_format(self):
        for prompt, expected, reply in [
            ("What is 7 + 8? Reply with only the number.", "15", "7 + 8 = 15"),
            ("What is 6 times 7? Reply with only the number.", "42", "6 \u00d7 7 = 42."),
        ]:
            with self.subTest(reply=reply):
                self.assertEqual(grade_canary(prompt, expected, reply), {"passed": True, "format_passed": False})
                self.assertEqual(grade_canary(prompt, expected, expected), {"passed": True, "format_passed": True})

    def test_wrong_equations_and_answer_mentions_fail(self):
        prompt = "What is 7 + 8? Reply with only the number."
        for reply in ("7 + 8 = 16", "1 + 14 = 15", "7 * 8 = 15", "The answer is not 15.",
                      "15 or 16", "7 + 8 = 15, actually 16", "7 + 8 = 15 = 16"):
            with self.subTest(reply=reply):
                self.assertFalse(grade_canary(prompt, "15", reply)["passed"])

    def test_non_arithmetic_checks_remain_exact(self):
        self.assertTrue(grade_canary("Copy exactly", "maple", "maple")["passed"])
        self.assertFalse(grade_canary("Copy exactly", "maple", "maple maple")["passed"])


if __name__ == "__main__":
    unittest.main()
