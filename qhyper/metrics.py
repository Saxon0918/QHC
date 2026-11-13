"""Candidate discrimination and exact-set hypergraph reconstruction metrics."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def canonical_edge(edge: Iterable[int]) -> tuple[int, ...]:
    """Normalize an edge to a sorted node tuple; reject duplicate members."""
    values = tuple(int(node) for node in edge)
    if len(values) < 2 or len(set(values)) != len(values) or any(node < 0 for node in values):
        raise ValueError("hyperedges need at least two unique nonnegative node IDs")
    return tuple(sorted(values))


def canonical_edge_set(edges: Iterable[Iterable[int]]) -> set[tuple[int, ...]]:
    """Deduplicate event multiplicity for the primary binary protocol."""
    return {canonical_edge(edge) for edge in edges}


def _candidate_arrays(
    labels: Iterable[int], scores: Iterable[float], sizes: Iterable[int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    truth = np.asarray(tuple(labels))
    score = np.asarray(tuple(scores), dtype=float)
    if truth.ndim != 1 or score.shape != truth.shape or not np.all(np.isin(truth, (0, 1))):
        raise ValueError("labels and scores must be aligned binary vectors")
    if not np.all(np.isfinite(score)) or np.any((score < 0) | (score > 1)):
        raise ValueError("scores must be finite probabilities in [0,1]")
    size_array = None
    if sizes is not None:
        size_array = np.asarray(tuple(sizes), dtype=int)
        if size_array.shape != truth.shape or np.any(size_array < 2):
            raise ValueError("sizes must align with candidates and be >=2")
    return truth.astype(int), score, size_array


def _candidate_summary(labels: np.ndarray, scores: np.ndarray, budget: int | None) -> dict:
    n = int(labels.size)
    positives = int(np.count_nonzero(labels))
    result: dict[str, float | int | None] = {
        "count": n,
        "positives": positives,
        "positive_rate": positives / n if n else None,
        "ap": None,
        "auroc": None,
        "brier": None,
        "log_loss": None,
        "precision_at_budget": None,
        "budget_used": None,
    }
    if n == 0:
        return result
    if positives:
        result["ap"] = float(average_precision_score(labels, scores))
    if 0 < positives < n:
        result["auroc"] = float(roc_auc_score(labels, scores))
    result["brier"] = float(brier_score_loss(labels, scores))
    result["log_loss"] = float(log_loss(labels, np.clip(scores, 1e-15, 1 - 1e-15), labels=[0, 1]))
    if budget is not None:
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        actual_budget = min(int(budget), n)
        result["budget_used"] = actual_budget
        if actual_budget:
            ranking = np.argsort(-scores, kind="mergesort")[:actual_budget]
            result["precision_at_budget"] = float(labels[ranking].mean())
    return result


def evaluate_candidate_predictions(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    sizes: Iterable[int] | None = None,
    budget: int | None = None,
) -> dict:
    """Evaluate one common candidate pool; undefined AP/AUROC are ``None``.

    AP is the primary imbalanced-data metric and is always accompanied by
    the positive base rate. A size-stratified summary is returned when sizes
    are available. Ties in Precision@Budget follow stable candidate order.
    """
    truth, score, size_array = _candidate_arrays(labels, scores, sizes)
    result = _candidate_summary(truth, score, budget)
    if size_array is not None:
        result["by_cardinality"] = {
            str(int(k)): _candidate_summary(truth[size_array == k], score[size_array == k], budget=None)
            for k in sorted(set(size_array.tolist()))
        }
    return result


def evaluate_hyperedge_sets(
    predicted_edges: Iterable[Iterable[int]], true_edges: Iterable[Iterable[int]]
) -> dict[str, int | float]:
    """Exact member-set metrics after deduplicating repeated events.

    Explicit empty-set convention: both sets empty score 1 on P/R/F1/J;
    only prediction empty gives all four 0; only truth empty gives P/F1/J=0
    and R=1 (no true edges were missed).
    """
    predicted = canonical_edge_set(predicted_edges)
    truth = canonical_edge_set(true_edges)
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    if not predicted and not truth:
        precision = recall = f1 = jaccard = 1.0
    else:
        precision = tp / (tp + fp) if predicted else 0.0
        recall = tp / (tp + fn) if truth else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        jaccard = tp / (tp + fp + fn)
    return {
        "predicted_count": len(predicted),
        "true_count": len(truth),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard),
    }


def candidate_recall(
    candidates: Iterable[Iterable[int]], true_edges: Iterable[Iterable[int]]
) -> dict:
    """Oracle recall upper bound for a fixed, deduplicated candidate pool."""
    candidate_set = canonical_edge_set(candidates)
    truth = canonical_edge_set(true_edges)
    covered = candidate_set & truth
    by_cardinality = {}
    for k in sorted({len(edge) for edge in truth}):
        truths_k = {edge for edge in truth if len(edge) == k}
        covered_k = truths_k & candidate_set
        by_cardinality[str(k)] = {
            "true_count": len(truths_k),
            "covered_count": len(covered_k),
            "recall": len(covered_k) / len(truths_k),
        }
    return {
        "candidate_count": len(candidate_set),
        "true_count": len(truth),
        "covered_count": len(covered),
        "recall": len(covered) / len(truth) if truth else 1.0,
        "by_cardinality": by_cardinality,
    }


def matched_edge_jaccard(
    predicted_edges: Iterable[Iterable[int]],
    true_edges: Iterable[Iterable[int]],
    *,
    minimum_jaccard: float = 0.0,
    max_edges: int = 2000,
) -> dict[str, float | int]:
    """Optional one-to-one approximate match; never replaces exact metrics."""
    if not 0 <= minimum_jaccard <= 1:
        raise ValueError("minimum_jaccard must be in [0,1]")
    predicted = sorted(canonical_edge_set(predicted_edges))
    truth = sorted(canonical_edge_set(true_edges))
    if max(len(predicted), len(truth)) > max_edges:
        raise ValueError("too many edges for dense Hungarian matching")
    if not predicted or not truth:
        return {"matched_count": 0, "mean_matched_jaccard": 0.0}
    weights = np.zeros((len(predicted), len(truth)), dtype=float)
    for i, edge_a in enumerate(predicted):
        set_a = set(edge_a)
        for j, edge_b in enumerate(truth):
            set_b = set(edge_b)
            weights[i, j] = len(set_a & set_b) / len(set_a | set_b)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    selected = [weights[i, j] for i, j in zip(rows, cols) if weights[i, j] >= minimum_jaccard]
    return {
        "matched_count": len(selected),
        "mean_matched_jaccard": float(np.mean(selected)) if selected else 0.0,
    }


def paired_bootstrap_delta(
    labels: Iterable[int],
    baseline_scores: Iterable[float],
    proposed_scores: Iterable[float],
    groups: Sequence[str | int],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    metric: str = "ap",
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    """Cluster-level paired bootstrap of proposed minus baseline performance.

    ``groups`` must identify independent graphs or time blocks. Resampling
    candidates individually would understate uncertainty when they share a
    projection graph. A positive delta means better performance, including
    for Brier and log loss (where we reverse the raw loss difference).
    """
    y, base, _ = _candidate_arrays(labels, baseline_scores)
    _, proposed, _ = _candidate_arrays(y, proposed_scores)
    group_array = np.asarray(tuple(groups))
    if group_array.shape != y.shape or y.size == 0:
        raise ValueError("groups must align with nonempty predictions")
    if n_bootstrap <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap parameters")
    if metric not in {"ap", "auroc", "brier", "log_loss"}:
        raise ValueError("unsupported metric")

    def delta(indices: np.ndarray) -> float | None:
        summary_a = _candidate_summary(y[indices], base[indices], None)
        summary_b = _candidate_summary(y[indices], proposed[indices], None)
        va, vb = summary_a[metric], summary_b[metric]
        if va is None or vb is None:
            return None
        return float(va - vb if metric in {"brier", "log_loss"} else vb - va)

    observed = delta(np.arange(y.size))
    unique_groups = np.unique(group_array)
    group_indices = [np.flatnonzero(group_array == group) for group in unique_groups]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, len(unique_groups), size=len(unique_groups))
        indices = np.concatenate([group_indices[i] for i in chosen])
        value = delta(indices)
        if value is not None:
            samples.append(value)
    if not samples:
        low = high = None
    else:
        alpha = (1 - confidence) / 2
        low, high = (float(x) for x in np.quantile(samples, [alpha, 1 - alpha]))
    return {
        "delta": observed,
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "bootstrap_valid": len(samples),
        "bootstrap_requested": n_bootstrap,
        "group_count": len(unique_groups),
    }
