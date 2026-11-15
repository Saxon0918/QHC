"""Greedy hard-hypergraph approximation of a graph-derived kernel.

For incidence ``B`` and nonnegative hyperedge weights ``w``, the normalized
operator is

    Theta = Dv^{-1/2} B diag(w_e / |e|) B.T Dv^{-1/2},
    Dv = diag(B w).

It is symmetric, positive semidefinite and entrywise nonnegative, with
``Theta sqrt(degree) = sqrt(degree)``. A quantum-walk probability kernel can
have negative eigenvalues, so exact approximation is not always possible.
The routine reports the actual residual; its greedy choices have no global
optimality guarantee. Unit-weight singleton edges keep every degree positive
and count toward the operator, but never toward the non-singleton budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse


Group = tuple[int, ...]


@dataclass(frozen=True)
class CompressionResult:
    """A fully specified incidence operator and its greedy search audit."""

    selected_groups: tuple[Group, ...]
    all_groups: tuple[Group, ...]
    weights: np.ndarray
    incidence: sparse.csr_matrix
    theta: np.ndarray
    error_fro: float
    history: tuple[dict[str, object], ...]
    status: str


def normalized_incidence_operator(
    incidence: sparse.spmatrix | np.ndarray,
    weights: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the exact normalized hard-incidence operator and node degrees."""
    B = sparse.csr_matrix(incidence, dtype=np.float64)
    if B.ndim != 2 or B.shape[0] == 0 or B.shape[1] == 0:
        raise ValueError("incidence must be a nonempty node-by-group matrix")
    B.sum_duplicates()
    B.eliminate_zeros()
    if B.data.size and (not np.isfinite(B.data).all() or
                        not np.all(B.data == 1.0)):
        raise ValueError("incidence must be binary")
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (B.shape[1],) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("weights must be finite nonnegative and column-aligned")
    sizes = np.asarray(B.sum(axis=0)).ravel()
    if np.any(sizes < 1):
        raise ValueError("every incidence column must contain at least one node")
    degrees = np.asarray(B @ w).ravel()
    if np.any(degrees <= 0):
        raise ValueError("every node must have positive weighted degree")
    numerator = (B @ sparse.diags(w / sizes, format="csr") @ B.T).toarray()
    inv_root = 1.0 / np.sqrt(degrees)
    theta = numerator * inv_root[:, None] * inv_root[None, :]
    return theta, degrees


def _checked_kernel(kernel: sparse.spmatrix | np.ndarray) -> np.ndarray:
    K = (kernel.toarray() if sparse.issparse(kernel) else np.asarray(kernel))
    if np.iscomplexobj(K) and np.any(np.imag(K) != 0):
        raise ValueError("kernel must be real")
    K = np.asarray(np.real(K), dtype=np.float64)
    if K.ndim != 2 or K.shape[0] == 0 or K.shape[0] != K.shape[1]:
        raise ValueError("kernel must be a nonempty square matrix")
    if not np.isfinite(K).all() or not np.allclose(K, K.T, atol=1e-10):
        raise ValueError("kernel must be finite and symmetric")
    return 0.5 * (K + K.T)


def _checked_groups(candidates: Iterable[Iterable[int]], n: int) -> tuple[Group, ...]:
    groups: set[Group] = set()
    for candidate in candidates:
        values = tuple(candidate)
        if (len(values) < 2 or len(values) > n or
                any(isinstance(node, (bool, np.bool_)) or
                    not isinstance(node, (int, np.integer)) for node in values)):
            raise ValueError("candidate groups must contain 2..n integer node IDs")
        group = tuple(sorted(int(node) for node in values))
        if len(set(group)) != len(group) or group[0] < 0 or group[-1] >= n:
            raise ValueError("candidate group contains repeated or invalid node IDs")
        groups.add(group)
    return tuple(sorted(groups))


def _trial_error_change(
    K: np.ndarray, theta: np.ndarray, degree: np.ndarray,
    nodes: np.ndarray, weight: float,
) -> float:
    """Exact squared-Frobenius change, touching only selected rows/columns."""
    updated_degree = degree[nodes] + weight
    row_scale = np.sqrt(degree[nodes] / updated_degree)
    new_rows = theta[nodes, :].copy() * row_scale[:, None]
    new_rows[:, nodes] *= row_scale[None, :]
    inverse_root = 1.0 / np.sqrt(updated_degree)
    new_rows[:, nodes] += (weight / len(nodes)) * (
        inverse_root[:, None] * inverse_root[None, :]
    )
    old_residual = theta[nodes, :] - K[nodes, :]
    new_residual = new_rows - K[nodes, :]
    change_rows = new_residual**2 - old_residual**2
    # Symmetry gives the same change in outside rows / selected columns.
    # The selected-by-selected block was counted twice and is subtracted once.
    return float(2.0 * change_rows.sum() - change_rows[:, nodes].sum())


def _add_group(
    theta: np.ndarray, degree: np.ndarray, nodes: np.ndarray, weight: float,
) -> None:
    """Update the full operator after accepting one candidate."""
    updated_degree = degree[nodes] + weight
    scale = np.ones(len(degree), dtype=np.float64)
    scale[nodes] = np.sqrt(degree[nodes] / updated_degree)
    theta *= scale[:, None] * scale[None, :]
    inverse_root = 1.0 / np.sqrt(updated_degree)
    theta[np.ix_(nodes, nodes)] += (weight / len(nodes)) * (
        inverse_root[:, None] * inverse_root[None, :]
    )
    degree[nodes] = updated_degree


def compress_kernel(
    K: sparse.spmatrix | np.ndarray,
    candidates: Iterable[Iterable[int]],
    *,
    budget: int,
    weight_grid: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    size_quota: Mapping[int, int] | None = None,
) -> CompressionResult:
    """Greedily reduce ``||K - Theta||_F`` using at most ``budget`` groups.

    At each step, all remaining candidate/positive-grid-weight pairs are
    evaluated against the *current normalized operator*. Previously chosen
    weights stay fixed. The best strictly improving pair is accepted; search
    stops early if none improves. Candidate and weight order resolve numerical
    ties reproducibly. An active budget with exact ties is not guaranteed to
    be permutation equivariant.
    """
    target = _checked_kernel(K)
    n = target.shape[0]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    grid = np.asarray(weight_grid, dtype=np.float64)
    if (grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all() or
            np.any(grid <= 0)):
        raise ValueError("weight_grid must contain finite positive weights")
    weights_to_try = tuple(float(w) for w in np.unique(grid))
    groups = _checked_groups(candidates, n)
    if size_quota is not None:
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 2 or
               isinstance(count, bool) or not isinstance(count, int) or count < 0
               for size, count in size_quota.items()):
            raise ValueError("size_quota must map sizes >= 2 to nonnegative integers")
        quotas = dict(size_quota)
    else:
        quotas = None

    theta = np.eye(n, dtype=np.float64)
    degree = np.ones(n, dtype=np.float64)
    current_error_sq = float(np.sum((target - theta)**2))
    selected: list[Group] = []
    chosen_weights: list[float] = []
    history: list[dict[str, object]] = []
    selected_set: set[Group] = set()
    chosen_by_size: dict[int, int] = {}
    status = "budget_exhausted" if budget == 0 else "candidate_exhausted"

    for _ in range(budget):
        best_group: Group | None = None
        best_weight = 0.0
        best_improvement = 0.0
        tolerance = 1e-12 * max(1.0, current_error_sq)
        any_eligible = False
        for group in groups:
            if group in selected_set:
                continue
            if quotas is not None and chosen_by_size.get(len(group), 0) >= quotas.get(len(group), 0):
                continue
            any_eligible = True
            nodes = np.fromiter(group, dtype=np.int64)
            for weight in weights_to_try:
                improvement = -_trial_error_change(
                    target, theta, degree, nodes, weight
                )
                if improvement > best_improvement + tolerance:
                    best_group = group
                    best_weight = weight
                    best_improvement = improvement
        if best_group is None:
            status = "no_improvement" if any_eligible else "candidate_exhausted"
            break
        nodes = np.fromiter(best_group, dtype=np.int64)
        old_error_sq = current_error_sq
        _add_group(theta, degree, nodes, best_weight)
        current_error_sq = float(np.sum((target - theta)**2))
        selected.append(best_group)
        chosen_weights.append(best_weight)
        selected_set.add(best_group)
        chosen_by_size[len(best_group)] = chosen_by_size.get(len(best_group), 0) + 1
        history.append({
            "group": best_group,
            "weight": best_weight,
            "error_fro": sqrt(current_error_sq),
            "squared_error_improvement": old_error_sq - current_error_sq,
        })
        status = "budget_exhausted"

    all_groups: tuple[Group, ...] = tuple((i,) for i in range(n)) + tuple(selected)
    all_weights = np.asarray([1.0] * n + chosen_weights, dtype=np.float64)
    rows = [node for group in all_groups for node in group]
    cols = [column for column, group in enumerate(all_groups) for _ in group]
    incidence = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(n, len(all_groups)),
    )
    # Re-evaluate from B and w to ensure the emitted artifact exactly follows
    # the stated normalization, independent of incremental search arithmetic.
    exact_theta, _ = normalized_incidence_operator(incidence, all_weights)
    return CompressionResult(
        selected_groups=tuple(selected),
        all_groups=all_groups,
        weights=all_weights,
        incidence=incidence,
        theta=exact_theta,
        error_fro=float(np.linalg.norm(target - exact_theta, ord="fro")),
        history=tuple(history),
        status=status,
    )
