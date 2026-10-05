import unittest
import numpy as np

from ai_happiness import corpus, joy_corpus
from ai_happiness.core import projection_diagnostic


class JoyDiagnosticTests(unittest.TestCase):
    def test_projection_auc_orientation_scale_and_ties(self):
        positive = [[2, 0], [3, 1]]
        neutral = [[0, 0], [1, 1]]
        result = projection_diagnostic(positive, neutral, [1, 0])
        self.assertEqual(result["auc"], 1)
        self.assertEqual(result["paired_win_fraction"], 1)
        self.assertEqual(result["mean_projection_gap"], 2)
        self.assertEqual(projection_diagnostic(positive, neutral, [10, 0]), result)
        self.assertEqual(projection_diagnostic(positive, neutral, [-1, 0])["auc"], 0)
        self.assertEqual(projection_diagnostic(positive, positive, [1, 0])["auc"], 0.5)

    def test_invalid_diagnostic_inputs_fail(self):
        for positive, neutral, vector in [
            ([], [], [1]), ([[1]], [[0]], [0]), ([[1]], [[0]], [1, 2]),
            ([[np.nan]], [[0]], [1]), ([[1]], [[np.inf]], [1]),
        ]:
            with self.subTest(positive=positive), self.assertRaises(ValueError):
                projection_diagnostic(positive, neutral, vector)

    def test_joy_sets_do_not_reuse_extraction_or_previous_validation_text(self):
        extraction = {text for pair in joy_corpus.PAIRS for text in pair}
        representation = {text for pair in joy_corpus.REPRESENTATION_PAIRS for text in pair}
        calibration = {text for probe in joy_corpus.CALIBRATION for text in probe}
        validation = {text for probe in joy_corpus.VALIDATION for text in probe}
        previous_validation = {text for probe in corpus.VALIDATION for text in probe}
        sets = [extraction, representation, calibration, validation, previous_validation]
        for index, texts in enumerate(sets):
            for other in sets[index + 1:]:
                self.assertFalse(texts & other)
        self.assertIs(joy_corpus.CANARIES, corpus.CANARIES)


if __name__ == "__main__":
    unittest.main()
