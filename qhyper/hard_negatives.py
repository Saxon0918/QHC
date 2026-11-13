"""Evaluation-only matching of positive candidates to hard negative cliques.

Call this module only after candidate generation, feature extraction, model
fitting and threshold selection are complete.  It is allowed to inspect held-
out hyperedge labels solely to define an *additional* matched diagnostic set.
The full candidate-pool AP remains the primary P1 outcome.  All input
candidates should already satisfy the same noiseless-projection clique rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score


_LEVELS = (
    "size_degree_source",
    "size_degree",
    "size_source",
    "size_only",
)


def _source_key(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ("unknown",)
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    if not values or any(not isinstance(item, str) or not item for item in values):
        raise ValueError("each proposal source must contain nonempty strings")
    return tuple(sorted(set(values)))


def _checked_inputs(
    labels: Sequence[int] | np.ndarray,
    sizes: Sequence[int] | np.ndarray,
    degree_means: Sequence[float] | np.ndarray,
    proposal_sources: Sequence[str | Iterable[str] | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, ...]]]:
    y = np.asarray(labels)
    size = np.asarray(sizes)
    degree = np.asarray(degree_means, dtype=np.float64)
    sources = [_source_key(value) for value in proposal_sources]
    if y.ndim != 1 or size.shape != y.shape or degree.shape != y.shape:
        raise ValueError("labels, sizes and degree_means must be aligned vectors")
    if len(sources) != y.size:
        raise ValueError("proposal_sources must align with candidate rows")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("labels must be binary")
    if np.any(size < 2) or not np.all(size == np.floor(size)):
        raise ValueError("candidate sizes must be integers >= 2")
    if not np.isfinite(degree).all() or np.any(degree < 0):
        raise ValueError("degree_means must be finite and nonnegative")
    return y.astype(np.int8), size.astype(np.int64), degree, sources


def match_hard_negatives_for_evaluation(
    labels: Sequence[int] | np.ndarray,
    sizes: Sequence[int] | np.ndarray,
    degree_means: Sequence[float] | np.ndarray,
    proposal_sources: Sequence[str | Iterable[str] | None],
    *,
    seed: int = 42,
    n_degree_bins: int = 4,
) -> dict[str, Any]:
    """Greedily pair each positive with one unused, size-matched negative.

    Degree bins are quantiles of the *evaluation candidate pool*.  The match
    first requires identical size, degree bin and proposal-source set; if
    unavailable it relaxes source, then degree bin, then both.  Size is never
    relaxed: a positive lacking an unused negative of the same size is omitted
    from the matched diagnostic and reported.  Within a level, larger overlap
    of proposal-source sets wins, followed by nearest degree mean and seeded
    random tie-breaking.  Rarer exact matches are processed first to preserve
    strong matches when possible.

    The function does not accept scores and cannot choose a model or threshold.
    The returned row indices are scoped to the original candidate order.
    """
    if not isinstance(n_degree_bins, int) or n_degree_bins < 1:
        raise ValueError("n_degree_bins must be a positive integer")
    y, size, degree, sources = _checked_inputs(
        labels, sizes, degree_means, proposal_sources
    )
    n = y.size
    rng = np.random.default_rng(seed)
    quantile_edges = (
        np.quantile(degree, np.linspace(0, 1, n_degree_bins + 1))
        if n else np.zeros(n_degree_bins + 1, dtype=np.float64)
    )
    degree_bin = np.searchsorted(quantile_edges[1:-1], degree, side="right")
    source_vocabulary = {key: index for index, key in enumerate(sorted(set(sources)))}
    source_code = np.asarray([source_vocabulary[key] for key in sources], dtype=int)

    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    available = np.zeros(n, dtype=bool)
    available[negatives] = True

    # The same seed and candidate rows produce exactly the same greedy order.
    priority_tie = rng.random(positives.size)
    priorities = []
    for ordinal, positive in enumerate(positives):
        exact_count = np.count_nonzero(
            (size[negatives] == size[positive])
            & (degree_bin[negatives] == degree_bin[positive])
            & (source_code[negatives] == source_code[positive])
        )
        same_size_count = np.count_nonzero(size[negatives] == size[positive])
        # Positives with a feasible exact match go first.  Among them, the
        # rarest exact matches have priority; zero-exact cases cannot consume
        # a negative needed for a stronger match before their fallback turn.
        priorities.append((exact_count == 0, exact_count, same_size_count,
                           priority_tie[ordinal], int(positive)))
    priorities.sort()

    pair_records: list[dict[str, Any]] = []
    unmatched: list[int] = []
    fallback_counts = {level: 0 for level in _LEVELS}
    for _, _, _, _, positive in priorities:
        same_size = available & (size == size[positive])
        conditions = (
            same_size & (degree_bin == degree_bin[positive])
            & (source_code == source_code[positive]),
            same_size & (degree_bin == degree_bin[positive]),
            same_size & (source_code == source_code[positive]),
            same_size,
        )
        chosen = None
        chosen_level = None
        for level, mask in zip(_LEVELS, conditions):
            pool = np.flatnonzero(mask)
            if pool.size == 0:
                continue
            positive_sources = set(sources[positive])
            source_overlap = np.asarray([
                len(positive_sources & set(sources[candidate]))
                / len(positive_sources | set(sources[candidate]))
                for candidate in pool
            ])
            pool = pool[np.isclose(source_overlap, source_overlap.max(), rtol=0, atol=1e-12)]
            distances = np.abs(degree[pool] - degree[positive])
            closest = pool[np.isclose(distances, distances.min(), rtol=0, atol=1e-12)]
            chosen = int(rng.choice(closest))
            chosen_level = level
            break
        if chosen is None:
            unmatched.append(positive)
            continue
        available[chosen] = False
        fallback_counts[chosen_level] += 1
        pair_records.append({
            "positive_index": positive,
            "negative_index": chosen,
            "match_level": chosen_level,
            "size": int(size[positive]),
            "positive_degree_bin": int(degree_bin[positive]),
            "negative_degree_bin": int(degree_bin[chosen]),
            "absolute_degree_mean_difference": float(abs(degree[positive] - degree[chosen])),
            "same_proposal_source": bool(source_code[positive] == source_code[chosen]),
            "proposal_source_jaccard": float(
                len(set(sources[positive]) & set(sources[chosen]))
                / len(set(sources[positive]) | set(sources[chosen]))
            ),
        })

    pair_records.sort(key=lambda record: record["positive_index"])
    matched_positive = [record["positive_index"] for record in pair_records]
    matched_negative = [record["negative_index"] for record in pair_records]
    by_size: dict[str, dict[str, int]] = {}
    for candidate_size in sorted(set(size[positives].tolist())):
        matched = sum(record["size"] == candidate_size for record in pair_records)
        total = int(np.count_nonzero(size[positives] == candidate_size))
        by_size[str(candidate_size)] = {
            "positive_count": total,
            "matched_pair_count": matched,
            "unmatched_positive_count": total - matched,
        }
    return {
        "method": "greedy_without_replacement_size_hard_constraint",
        "seed": int(seed),
        "n_degree_bins": int(n_degree_bins),
        "degree_quantile_edges": quantile_edges.tolist(),
        "candidate_count": int(n),
        "positive_count": int(positives.size),
        "negative_count": int(negatives.size),
        "matched_pair_count": len(pair_records),
        "matched_positive_fraction": len(pair_records) / positives.size if positives.size else None,
        "positive_indices": matched_positive,
        "negative_indices": matched_negative,
        "unmatched_positive_indices": sorted(unmatched),
        "fallback_counts": fallback_counts,
        "by_size": by_size,
        "pair_records": pair_records,
    }


def matched_average_precision(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    matching: dict[str, Any],
) -> dict[str, float | int | None]:
    """Evaluate AP on predeclared matched rows; never replace full-pool AP."""
    y = np.asarray(labels)
    score = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or score.shape != y.shape:
        raise ValueError("labels and scores must be aligned vectors")
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("scores must be finite probabilities in [0,1]")
    positive = np.asarray(matching["positive_indices"], dtype=int)
    negative = np.asarray(matching["negative_indices"], dtype=int)
    if positive.shape != negative.shape:
        raise ValueError("matching must contain equal positive and negative counts")
    if (np.any(positive < 0) or np.any(negative < 0)
            or np.any(positive >= y.size) or np.any(negative >= y.size)):
        raise ValueError("matched index is outside the candidate rows")
    if len(set(positive.tolist() + negative.tolist())) != 2 * positive.size:
        raise ValueError("matched indices must be unique")
    if not np.all(y[positive] == 1) or not np.all(y[negative] == 0):
        raise ValueError("matching indices disagree with supplied evaluation labels")
    if positive.size == 0:
        return {"matched_ap": None, "matched_pair_count": 0,
                "matched_candidate_count": 0, "matched_positive_rate": None}
    selected = np.concatenate((positive, negative))
    return {
        "matched_ap": float(average_precision_score(y[selected], score[selected])),
        "matched_pair_count": int(positive.size),
        "matched_candidate_count": int(selected.size),
        "matched_positive_rate": float(np.mean(y[selected])),
    }
