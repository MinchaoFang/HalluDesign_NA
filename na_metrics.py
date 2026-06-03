from __future__ import annotations

from typing import Any


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "mean") and not isinstance(value, (float, int)):
        try:
            value = value.mean()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        return None


def best_summary_index(results: dict[str, Any]) -> int:
    summaries = results.get("summary_confidence", [])
    if not summaries:
        return 0
    best_idx = 0
    best_score = float("-inf")
    for idx, summary in enumerate(summaries):
        score = to_float(summary.get("ranking_score"))
        if score is not None and score > best_score:
            best_idx = idx
            best_score = score
    return best_idx


def chain_pair_mean(
    matrix: Any,
    protein_indices: list[int],
    other_indices: list[int],
) -> float | None:
    values = []
    for i in protein_indices:
        for j in other_indices:
            for a, b in ((i, j), (j, i)):
                try:
                    value = matrix[a][b]
                except Exception:
                    value = None
                value = to_float(value)
                if value is not None:
                    values.append(value)
    if not values:
        return None
    return sum(values) / len(values)
