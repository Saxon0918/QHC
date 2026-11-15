"""Leakage-controlled node-task evaluation for constructed hypergraphs.

Every arm receives the same node features, split and logistic-regression
decoder.  Only its supplied propagation operator changes.  The complete
graph/operator may contain unlabeled test nodes (a transductive setting), but
test *labels* are not read until the final metrics are calculated.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Iterable, Mapping

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler


_DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)


def _validated_operator(
    theta: sparse.spmatrix | np.ndarray,
) -> sparse.csr_matrix | np.ndarray:
    """Validate an operator without materializing a sparse N-by-N matrix."""
    if sparse.issparse(theta):
        if np.iscomplexobj(theta.data):
            raise ValueError("propagation operator must be real")
        try:
            operator = theta.tocsr(copy=True).astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("propagation operator must be numeric") from exc
        operator.sum_duplicates()
        operator.eliminate_zeros()
        if (
            operator.ndim != 2
            or operator.shape[0] != operator.shape[1]
            or operator.shape[0] == 0
            or not np.isfinite(operator.data).all()
        ):
            raise ValueError("propagation operator must be finite and square")
        return operator
    raw = np.asarray(theta)
    if np.iscomplexobj(raw):
        raise ValueError("propagation operator must be real")
    try:
        operator = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("propagation operator must be numeric") from exc
    if (
        operator.ndim != 2
        or operator.shape[0] != operator.shape[1]
        or operator.shape[0] == 0
        or not np.isfinite(operator).all()
    ):
        raise ValueError("propagation operator must be finite and square")
    return operator


def _dense_features(features: sparse.spmatrix | np.ndarray, n: int) -> np.ndarray:
    raw = features.toarray() if sparse.issparse(features) else np.asarray(features)
    if np.iscomplexobj(raw):
        raise ValueError("features must be real")
    try:
        x = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be numeric") from exc
    if x.ndim != 2 or x.shape[0] != n or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("features must be a finite N x D matrix")
    return x


def _validated_split(split: Mapping[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    if not isinstance(split, Mapping) or not {"train", "validation", "test"}.issubset(split):
        raise ValueError("split needs train, validation, and test indices")
    indices: dict[str, np.ndarray] = {}
    for role in ("train", "validation", "test"):
        raw = np.asarray(split[role])
        if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"{role} indices must be a nonempty integer array")
        if np.any(raw < 0) or np.any(raw >= n) or np.unique(raw).size != raw.size:
            raise ValueError(f"{role} indices are out of range or repeated")
        indices[role] = raw.astype(np.int64, copy=False)
    concatenated = np.concatenate(tuple(indices.values()))
    if np.unique(concatenated).size != concatenated.size:
        raise ValueError("train, validation, and test indices must be disjoint")
    return indices


def _validated_grid(c_grid: Iterable[float]) -> list[float]:
    try:
        values = list(c_grid)
        grid = sorted({float(value) for value in values})
    except (TypeError, ValueError) as exc:
        raise ValueError("C_grid must contain positive finite numbers") from exc
    if not grid or any(not np.isfinite(value) or value <= 0.0 for value in grid):
        raise ValueError("C_grid must contain positive finite numbers")
    return grid


def _model_fingerprint(
    scaler: StandardScaler, model: LogisticRegression, selected_c: float
) -> str:
    """Stable audit token for checking that held-out labels did not alter fit."""
    payload = {
        "selected_C": selected_c,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "classes": model.classes_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


def evaluate_operator(
    theta: sparse.spmatrix | np.ndarray,
    X: sparse.spmatrix | np.ndarray,
    y: np.ndarray,
    split: Mapping[str, np.ndarray],
    C_grid: Iterable[float] = _DEFAULT_C_GRID,
) -> dict[str, float | int | str]:
    """Fit one fixed linear decoder after ``theta @ X`` propagation.

    Standardization and logistic regression are fitted on training nodes only.
    The regularization ``C`` minimizing validation log loss is selected, with
    smaller ``C`` breaking exact ties.  The selected *training-fitted* model is
    evaluated once on test nodes; validation nodes are not used for refitting.
    The returned fingerprint makes test-label permutation leakage testable.
    """
    operator = _validated_operator(theta)
    n = operator.shape[0]
    features = _dense_features(X, n)
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.shape[0] != n:
        raise ValueError("labels must be a length-N vector")
    roles = _validated_split(split, n)
    grid = _validated_grid(C_grid)

    train = roles["train"]
    validation = roles["validation"]
    test = roles["test"]
    train_classes = np.unique(labels[train])
    if train_classes.size < 2:
        raise ValueError("training split must contain at least two classes")
    if not np.isin(labels[validation], train_classes).all():
        raise ValueError("validation labels contain a class absent from training")

    propagated = operator @ features
    scaler = StandardScaler().fit(propagated[train])
    train_x = scaler.transform(propagated[train])
    validation_x = scaler.transform(propagated[validation])
    best: tuple[float, float, LogisticRegression] | None = None
    for c in grid:
        model = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=0)
        model.fit(train_x, labels[train])
        validation_probabilities = model.predict_proba(validation_x)
        validation_loss = float(
            log_loss(labels[validation], validation_probabilities, labels=model.classes_)
        )
        if best is None or (validation_loss, c) < (best[0], best[1]):
            best = validation_loss, c, model
    assert best is not None  # A nonempty grid was validated above.
    validation_loss, selected_c, selected_model = best

    # Held-out labels first enter the computation at these final metrics.
    test_x = scaler.transform(propagated[test])
    test_probabilities = selected_model.predict_proba(test_x)
    test_predictions = selected_model.predict(test_x)
    validation_predictions = selected_model.predict(validation_x)
    if not np.isin(labels[test], selected_model.classes_).all():
        raise ValueError("test labels contain a class absent from training")
    return {
        "selected_C": float(selected_c),
        "validation_log_loss": float(validation_loss),
        "validation_macro_f1": float(
            f1_score(
                labels[validation], validation_predictions,
                labels=selected_model.classes_, average="macro", zero_division=0,
            )
        ),
        "test_macro_f1": float(
            f1_score(
                labels[test], test_predictions,
                labels=selected_model.classes_, average="macro", zero_division=0,
            )
        ),
        "test_log_loss": float(
            log_loss(labels[test], test_probabilities, labels=selected_model.classes_)
        ),
        "test_accuracy": float(accuracy_score(labels[test], test_predictions)),
        "train_size": int(train.size),
        "validation_size": int(validation.size),
        "test_size": int(test.size),
        "model_fingerprint": _model_fingerprint(scaler, selected_model, selected_c),
    }


def fused_operator(
    graph: sparse.spmatrix | np.ndarray,
    additional: sparse.spmatrix | np.ndarray,
    alpha: float,
) -> sparse.csr_matrix | np.ndarray:
    """Convexly combine graph and additional propagation operators.

    The additional operator may come from a soft hypergraph or be the identity
    control.  Endpoints return the corresponding operator exactly, and sparse
    inputs stay sparse so the large-graph path never forms an N-by-N array.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) \
            or not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")
    base = _validated_operator(graph)
    other = _validated_operator(additional)
    if base.shape != other.shape:
        raise ValueError("fusion operators must have the same shape")
    if alpha == 0.0:
        return base
    if alpha == 1.0:
        return other
    if sparse.issparse(base) or sparse.issparse(other):
        first = base if sparse.issparse(base) else sparse.csr_matrix(base)
        second = other if sparse.issparse(other) else sparse.csr_matrix(other)
        fused = ((1.0 - alpha) * first + alpha * second).tocsr()
        fused.eliminate_zeros()
        return fused
    return (1.0 - alpha) * base + alpha * other


def evaluate_fusion(
    graph: sparse.spmatrix | np.ndarray,
    additional: sparse.spmatrix | np.ndarray,
    X: sparse.spmatrix | np.ndarray,
    y: np.ndarray,
    split: Mapping[str, np.ndarray],
    alphas: Iterable[float],
    C_grid: Iterable[float] = _DEFAULT_C_GRID,
) -> dict[str, object]:
    """Choose fusion weight and decoder using validation labels only.

    Every alpha gets the same train-only scaler and C search.  Test labels are
    read once, after the lowest-validation-loss (alpha, C) pair is selected.
    The complete validation curve is retained for auditing without exposing
    test scores for unselected alpha values.
    """
    base = _validated_operator(graph)
    other = _validated_operator(additional)
    if base.shape != other.shape:
        raise ValueError("fusion operators must have the same shape")
    try:
        alpha_values = list(alphas)
    except TypeError as exc:
        raise ValueError("alphas must be a nonempty sequence in [0, 1]") from exc
    if (not alpha_values or any(isinstance(a, bool) or not isinstance(a, (int, float))
                                or not np.isfinite(a) or not 0.0 <= a <= 1.0
                                for a in alpha_values)
            or len(set(alpha_values)) != len(alpha_values)):
        raise ValueError("alphas must be distinct finite values in [0, 1]")
    alpha_values = sorted(float(a) for a in alpha_values)
    grid = _validated_grid(C_grid)
    n = base.shape[0]
    features = _dense_features(X, n)
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.shape[0] != n:
        raise ValueError("labels must be a length-N vector")
    roles = _validated_split(split, n)
    train, validation, test = (roles[role] for role in ("train", "validation", "test"))
    train_classes = np.unique(labels[train])
    if train_classes.size < 2:
        raise ValueError("training split must contain at least two classes")
    if not np.isin(labels[validation], train_classes).all():
        raise ValueError("validation labels contain a class absent from training")

    validation_by_alpha: list[dict[str, float]] = []
    selected: tuple[tuple[float, float, float], StandardScaler,
                    LogisticRegression, np.ndarray] | None = None
    for alpha in alpha_values:
        propagated = fused_operator(base, other, alpha) @ features
        scaler = StandardScaler().fit(propagated[train])
        train_x = scaler.transform(propagated[train])
        validation_x = scaler.transform(propagated[validation])
        best_for_alpha: tuple[float, float, LogisticRegression] | None = None
        for c in grid:
            model = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=0)
            model.fit(train_x, labels[train])
            probability = model.predict_proba(validation_x)
            loss = float(log_loss(labels[validation], probability, labels=model.classes_))
            if best_for_alpha is None or (loss, c) < best_for_alpha[:2]:
                best_for_alpha = (loss, c, model)
        assert best_for_alpha is not None
        loss, c, model = best_for_alpha
        validation_by_alpha.append({
            "alpha": alpha,
            "selected_C": c,
            "validation_log_loss": loss,
            "validation_macro_f1": float(f1_score(
                labels[validation], model.predict(validation_x),
                labels=model.classes_, average="macro", zero_division=0)),
        })
        key = (loss, alpha, c)
        if selected is None or key < selected[0]:
            selected = (key, scaler, model, propagated)
    assert selected is not None
    (validation_loss, selected_alpha, selected_c), scaler, model, propagated = selected
    if not np.isin(labels[test], model.classes_).all():
        raise ValueError("test labels contain a class absent from training")
    test_x = scaler.transform(propagated[test])
    test_probabilities = model.predict_proba(test_x)
    validation_x = scaler.transform(propagated[validation])
    return {
        "selected_alpha": selected_alpha,
        "alpha_grid": alpha_values,
        "validation_by_alpha": validation_by_alpha,
        "selected_C": selected_c,
        "validation_log_loss": validation_loss,
        "validation_macro_f1": float(f1_score(
            labels[validation], model.predict(validation_x),
            labels=model.classes_, average="macro", zero_division=0)),
        "test_macro_f1": float(f1_score(
            labels[test], model.predict(test_x), labels=model.classes_,
            average="macro", zero_division=0)),
        "test_log_loss": float(log_loss(
            labels[test], test_probabilities, labels=model.classes_)),
        "test_accuracy": float(accuracy_score(labels[test], model.predict(test_x))),
        "train_size": int(train.size),
        "validation_size": int(validation.size),
        "test_size": int(test.size),
        "model_fingerprint": _model_fingerprint(scaler, model, selected_c),
    }


def operator_equivalence_error(
    left: sparse.spmatrix | np.ndarray,
    right: sparse.spmatrix | np.ndarray,
) -> float:
    """Maximum entrywise difference between two linear propagation operators."""
    first = _validated_operator(left)
    second = _validated_operator(right)
    if first.shape != second.shape:
        raise ValueError("propagation operators must have the same shape")
    if sparse.issparse(first) or sparse.issparse(second):
        sparse_first = first if sparse.issparse(first) else sparse.csr_matrix(first)
        sparse_second = second if sparse.issparse(second) else sparse.csr_matrix(second)
        difference = sparse_first - sparse_second
        difference.sum_duplicates()
        difference.eliminate_zeros()
        return float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    return float(np.max(np.abs(first - second)))


def assert_operator_equivalence(
    left: sparse.spmatrix | np.ndarray,
    right: sparse.spmatrix | np.ndarray,
    atol: float = 1e-10,
) -> float:
    """Check the graph/hypergraph linear-operator equivalence directly.

    Equality of downstream metrics on one task would be weaker evidence: two
    distinct operators can act alike on one particular feature matrix.
    """
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    error = operator_equivalence_error(left, right)
    if error > atol:
        raise AssertionError(f"propagation operators differ (max error={error:.3g})")
    return error
