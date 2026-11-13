"""Train-only candidate scoring and validation-only hyperedge selection."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qhyper.metrics import canonical_edge


def _feature_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must be finite")
    return matrix


def _binary_labels(labels: Iterable[int], n: int) -> np.ndarray:
    labels = np.asarray(tuple(labels))
    if labels.shape != (n,) or not np.all(np.isin(labels, (0, 1))):
        raise ValueError("labels must be a binary vector aligned with features")
    return labels.astype(np.int64)


def fit_candidate_scorer(
    train_features: np.ndarray,
    train_labels: Iterable[int],
    *,
    C: float = 1.0,
    class_weight: str | dict[int, float] | None = "balanced",
    random_state: int = 42,
) -> Pipeline:
    """Fit the shared StandardScaler + LogisticRegression on *training* rows.

    Feature scaling and class weighting are estimated solely from train data.
    The caller must not pass validation or test rows to this function.
    """
    matrix = _feature_matrix(train_features)
    labels = _binary_labels(train_labels, matrix.shape[0])
    if np.unique(labels).size < 2:
        raise ValueError("training labels must contain both classes")
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be positive and finite")
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(C),
                    class_weight=class_weight,
                    max_iter=2000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(matrix, labels)
    return model


def predict_candidate_scores(model: Pipeline, features: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities for held-out candidates."""
    matrix = _feature_matrix(features)
    return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)


def choose_validation_threshold(
    validation_labels: Iterable[int], validation_scores: Iterable[float]
) -> dict[str, float | int]:
    """Select the compactness-tied F1 threshold using validation only.

    We test every distinct score as an inclusive threshold and an all-empty
    option. Tied F1 values choose the *highest* threshold, a deterministic
    preference for fewer predicted edges. Test truth must never be supplied.
    """
    scores = np.asarray(tuple(validation_scores), dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("validation_scores must be a nonempty finite vector")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("validation_scores must be probabilities in [0,1]")
    labels = _binary_labels(validation_labels, scores.size)
    thresholds = np.unique(scores)
    empty_threshold = float(np.nextafter(np.max(scores), np.inf))
    thresholds = np.append(thresholds, empty_threshold)
    best = (-1.0, -1.0, 0)
    for threshold in thresholds:
        pred = scores >= threshold
        score = float(f1_score(labels, pred, zero_division=0))
        key = (score, float(threshold), int(np.count_nonzero(pred)))
        if key[:2] > best[:2]:
            best = key
    return {"threshold": best[1], "validation_f1": best[0], "selected_count": best[2]}


def select_hyperedges(
    candidates: Iterable[Iterable[int]], scores: Iterable[float], threshold: float
) -> list[tuple[int, ...]]:
    """Select canonical, deduplicated candidates at a fixed threshold."""
    candidate_list = [canonical_edge(edge) for edge in candidates]
    scores_array = np.asarray(tuple(scores), dtype=np.float64)
    if scores_array.shape != (len(candidate_list),):
        raise ValueError("scores must align with candidates")
    if not np.isfinite(threshold) or not np.all(np.isfinite(scores_array)):
        raise ValueError("threshold and scores must be finite")
    if np.any((scores_array < 0) | (scores_array > 1)):
        raise ValueError("scores must be probabilities in [0,1]")
    return sorted({edge for edge, score in zip(candidate_list, scores_array) if score >= threshold})
