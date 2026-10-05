"""Numerics and selection policy, usable without PyTorch or model weights."""
import math
import re
import numpy as np


def nonnegative(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Dose must be finite and nonnegative.")
    return value


def fit_direction(positive, neutral, denoise=0.5):
    """Difference of means, optionally removing neutral PCs by variance share.

    The returned delta retains its magnitude: dose 1 adds one denoised
    positive-minus-neutral mean difference. This is not the upstream dose unit.
    """
    p = np.asarray(positive, dtype=np.float64)
    n = np.asarray(neutral, dtype=np.float64)
    if p.ndim != 2 or n.shape != p.shape or len(p) < 2:
        raise ValueError("Expected matching activation matrices with at least two rows.")
    if not np.isfinite(p).all() or not np.isfinite(n).all():
        raise ValueError("Nonfinite extraction activations.")
    if not math.isfinite(denoise) or not 0 <= denoise < 1:
        raise ValueError("Denoise must be in [0, 1).")
    raw = p.mean(axis=0) - n.mean(axis=0)
    delta = raw.copy()
    removed = 0
    if denoise:
        _, singular, vh = np.linalg.svd(n - n.mean(axis=0), full_matrices=False)
        variance = singular ** 2
        if variance.sum() > 1e-12:
            removed = int(np.searchsorted(np.cumsum(variance) / variance.sum(), denoise) + 1)
            basis = vh[:removed]
            delta -= basis.T @ (basis @ delta)
    norm = float(np.linalg.norm(delta))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("Positive contrast vanished; use more examples or less denoising.")
    return delta.astype(np.float32), {
        "raw_norm": float(np.linalg.norm(raw)), "delta_norm": norm,
        "removed_neutral_pcs": removed,
        "training_projection_gap": float(raw @ (delta / norm)),
    }


def text_metrics(text):
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    trigrams = list(zip(words, words[1:], words[2:]))
    return {
        "words": len(words),
        "distinct_word_ratio": len(set(words)) / max(1, len(words)),
        "repeated_trigram_fraction": 1 - len(set(trigrams)) / max(1, len(trigrams))
        if trigrams else 0.0,
    }


def projection_diagnostic(positive, control, vector):
    """Held-out sentence projections; semantic separation, not felt valence."""
    p = np.asarray(positive, dtype=np.float64)
    c = np.asarray(control, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    if (p.ndim != 2 or p.shape != c.shape or not len(p) or v.ndim != 1
            or p.shape[1] != len(v) or not all(np.isfinite(x).all() for x in (p, c, v))):
        raise ValueError("Expected finite matching activation matrices and a direction.")
    norm = float(np.linalg.norm(v))
    if norm <= 0:
        raise ValueError("Direction must be nonzero.")
    positive_scores, control_scores = p @ (v / norm), c @ (v / norm)
    difference = positive_scores[:, None] - control_scores[None, :]
    return {
        "pairs": len(p),
        "positive_projections": positive_scores.tolist(),
        "control_projections": control_scores.tolist(),
        "mean_projection_gap": float(np.mean(positive_scores - control_scores)),
        "paired_win_fraction": float(np.mean(positive_scores > control_scores)),
        "auc": float(np.mean((difference > 0) + 0.5 * (difference == 0))),
        "interpretation": "Separation of held-out sentence representations, not subjective experience.",
    }


def grade_canary(prompt, expected, text):
    """Separate a correct arithmetic equation from exact answer formatting.

    Accept only the requested operands, operation, and correct result. Never
    look for an answer substring: contradictory or unrelated text must fail.
    Non-arithmetic canaries still require an exact answer.
    """
    answer = text.strip().rstrip(".! ")
    exact = answer.casefold() == expected.casefold()
    passed = exact
    question = re.fullmatch(r"What is (\d+) (\+|times) (\d+)\? Reply with only the number\.", prompt)
    equation = re.fullmatch(r"(\d+)\s*(\+|[xX\u00d7*])\s*(\d+)\s*=\s*(\d+)", answer)
    if question and equation and expected.isdecimal():
        left, right = int(question[1]), int(question[3])
        operation = "+" if question[2] == "+" else "*"
        actual_operation = "+" if equation[2] == "+" else "*"
        result = left + right if operation == "+" else left * right
        passed = (int(equation[1]) == left and int(equation[3]) == right
                  and actual_operation == operation and int(equation[4]) == int(expected) == result)
    return {"passed": passed, "format_passed": exact}


def quality_failures(cell, baseline, max_repetition=0.18, min_distinct=0.30):
    failures = []
    if not math.isfinite(cell["positive_score"]):
        failures.append("nonfinite positive-language score")
    for index, sample in enumerate(cell["samples"]):
        m = sample["metrics"]
        if m["words"] < 12:
            failures.append(f"sample {index}: too short for coherence screening")
        if m["repeated_trigram_fraction"] > max_repetition:
            failures.append(f"sample {index}: repeated phrases")
        if m["distinct_word_ratio"] < min_distinct:
            failures.append(f"sample {index}: low word diversity")
    if len(cell["canaries"]) != len(baseline["canaries"]):
        raise ValueError("Canary counts do not match.")
    for index, (current, original) in enumerate(zip(cell["canaries"], baseline["canaries"])):
        if original["passed"] and not current["passed"]:
            failures.append(f"canary {index}: lost a baseline capability")
        if original.get("format_passed", False) and not current.get("format_passed", False):
            failures.append(f"canary {index}: lost baseline answer formatting")
    if sum(x["passed"] for x in cell["canaries"]) < 3:
        failures.append("fewer than three capability checks passed")
    return failures


def select_best(cells, baseline, min_gain=0.02):
    """Maximize the measured score among eligible tested cells; abstain on nulls."""
    if not math.isfinite(min_gain) or min_gain <= 0:
        raise ValueError("Minimum gain must be finite and positive.")
    if not math.isfinite(baseline["positive_score"]):
        raise ValueError("Baseline score is nonfinite.")
    candidates = [c for c in cells if nonnegative(c["dose"]) > 0
                  and not c["failures"] and math.isfinite(c["positive_score"])
                  and c["positive_score"] - baseline["positive_score"] >= min_gain]
    return max(candidates, key=lambda c: (c["positive_score"], -c["dose"])) if candidates else None
