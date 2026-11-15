"""Small-graph transport kernels for the latent-hypergraph construction study.

These functions deliberately compute the full matrix exponential.  The hard
node cap prevents a dense ``N x N`` allocation from silently becoming a
large-graph experiment.  Both quantum and classical controls use only the
same supplied real symmetric Hamiltonian; neither sees hyperedge truth or
node labels.

For ``U_t = exp(-i t H)`` and ``P_t = abs(U_t)**2``, each column of
``B_t = P_t.T`` is a soft hyperedge corresponding to one measured endpoint.
Because ``P_t`` is doubly stochastic, both node and hyperedge degrees equal
one.  The standard linear HGNN operator is therefore exactly
``B_t @ B_t.T = P_t.T @ P_t``.  This operator also is an ordinary weighted
graph kernel, so downstream gains alone cannot establish irreducible
higher-order interactions.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.sparse.csgraph import connected_components


def _validated_inputs(
    hamiltonian: sparse.spmatrix | np.ndarray,
    times: Iterable[float],
    weights: Iterable[float] | None,
    max_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    shape = getattr(hamiltonian, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1] or shape[0] < 1:
        raise ValueError("Hamiltonian must be a nonempty square matrix")
    if shape[0] > max_nodes:
        raise ValueError(
            f"dense transport is limited to max_nodes={max_nodes}; "
            f"received {shape[0]} nodes"
        )
    raw = hamiltonian.toarray() if sparse.issparse(hamiltonian) else np.asarray(hamiltonian)
    if np.iscomplexobj(raw):
        raise ValueError("Hamiltonian must be real")
    try:
        h = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Hamiltonian must contain real numeric values") from exc
    if not np.isfinite(h).all():
        raise ValueError("Hamiltonian contains nonfinite values")
    if not np.allclose(h, h.T, rtol=0.0, atol=1e-12):
        raise ValueError("Hamiltonian must be symmetric")

    try:
        t = np.asarray(list(times), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("times must be a nonempty sequence of real values") from exc
    if t.ndim != 1 or t.size == 0 or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("times must be a nonempty sequence of finite nonnegative values")

    if weights is None:
        w = np.full(t.size, 1.0 / t.size, dtype=np.float64)
    else:
        try:
            w = np.asarray(list(weights), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("weights must be real values") from exc
        if (w.ndim != 1 or w.size != t.size or not np.isfinite(w).all()
                or np.any(w < 0) or float(w.sum()) <= 0):
            raise ValueError("weights must match times, be nonnegative, and have positive sum")
        w = w / w.sum()
    return h, t, w


def _transport_result(
    matrices: list[np.ndarray], times: np.ndarray, weights: np.ndarray
) -> dict[str, object]:
    n = matrices[0].shape[0]
    kernel = np.zeros((n, n), dtype=np.float64)
    for weight, matrix in zip(weights, matrices):
        kernel += float(weight) * (matrix.T @ matrix)
    return {
        "transport": matrices,
        "K": kernel,
        "times": times.copy(),
        "weights": weights.copy(),
    }


def _matched_generator(h: np.ndarray) -> np.ndarray:
    off_diagonal = h * h
    np.fill_diagonal(off_diagonal, 0.0)
    return off_diagonal - np.diag(off_diagonal.sum(axis=1))


def _stationary_projector(generator: np.ndarray) -> tuple[int, np.ndarray]:
    """Uniform-within-component limit of ``exp(s Q)`` as ``s -> infinity``."""
    connectivity = generator.copy()
    np.fill_diagonal(connectivity, 0.0)
    count, labels = connected_components(
        sparse.csr_matrix(connectivity), directed=False
    )
    projector = np.zeros_like(generator)
    for component in range(count):
        members = np.flatnonzero(labels == component)
        projector[np.ix_(members, members)] = 1.0 / len(members)
    return count, projector


def quantum_transport(
    hamiltonian: sparse.spmatrix | np.ndarray,
    times: Iterable[float],
    weights: Iterable[float] | None = None,
    max_nodes: int = 128,
) -> dict[str, object]:
    """Return ``P_t=|exp(-itH)|²`` and ``K=Σ_t ω_t P_t.T P_t``.

    Nonnegative weights are normalized to sum to one.  For real symmetric
    ``H``, ``P_t`` is symmetric and doubly stochastic.  ``K`` is symmetric,
    positive semidefinite and doubly stochastic; it is *not* evidence that
    hidden event hyperedges have been recovered.
    """
    h, t, w = _validated_inputs(hamiltonian, times, weights, max_nodes)
    matrices = [np.abs(expm(-1j * float(time) * h)) ** 2 for time in t]
    return _transport_result(matrices, t, w)


def matched_markov(
    hamiltonian: sparse.spmatrix | np.ndarray,
    times: Iterable[float],
    weights: Iterable[float] | None = None,
    max_nodes: int = 128,
) -> dict[str, object]:
    """Classical transport matched to the quantum kernel's short-time term.

    Set ``Q_ij=H_ij²`` for ``i!=j`` and ``Q_ii=-Σ_{j!=i}Q_ij``.  Then
    ``C_t=exp(t² Q)`` is a symmetric Markov kernel.  Quantum transport obeys
    ``P_t=I+t² Q+O(t⁴)``, hence both transport matrices and their Gram kernels
    agree through order ``t²``.  The returned classical ``K`` is exactly
    ``Σ_t ω_t exp(2t² Q)``, not merely a short-time approximation to it.
    """
    h, t, w = _validated_inputs(hamiltonian, times, weights, max_nodes)
    generator = _matched_generator(h)
    matrices = [expm(float(time) ** 2 * generator) for time in t]
    result = _transport_result(matrices, t, w)
    result["generator"] = generator
    return result


def spread_matched_markov(
    hamiltonian: sparse.spmatrix | np.ndarray,
    times: Iterable[float],
    weights: Iterable[float] | None = None,
    max_nodes: int = 128,
) -> dict[str, object]:
    """Match each quantum transport kernel's trace with a classical kernel.

    For each requested quantum time ``t``, solve for ``s >= 0`` such that
    ``tr(exp(2sQ)) = tr(P_t.T P_t)``, where ``Q_ij=H_ij²`` off the diagonal
    and row sums of ``Q`` vanish.  The left side decreases strictly from
    ``n`` to the number of connected components ``c`` unless ``Q=0``.
    A target exactly at the limiting value ``c`` is represented by
    ``s=infinity`` and the blockwise-uniform stationary projector.  This
    controls total mixing at finite quantum time; it does not match the full
    spectrum or constitute a causal attribution of any performance gap.

    Returns the usual ``transport``, ``K``, ``times``, ``weights`` fields,
    plus ``markov_times`` (the solved ``s`` values), ``generator``, and
    ``target_traces``/``achieved_traces`` for an auditable numerical check.
    """
    h, t, w = _validated_inputs(hamiltonian, times, weights, max_nodes)
    generator = _matched_generator(h)
    components, stationary = _stationary_projector(generator)
    eigenvalues = np.linalg.eigvalsh(generator)
    # The largest ``components`` eigenvalues are the exact zero modes.
    # Numerical eigensolvers may return them as tiny signed values.
    decay_rates = np.maximum(-2.0 * eigenvalues[:h.shape[0] - components], 0.0)
    matrices: list[np.ndarray] = []
    markov_times: list[float] = []
    target_traces: list[float] = []
    achieved_traces: list[float] = []
    n = h.shape[0]
    tolerance = 1e-10 * max(1, n)
    for time in t:
        quantum = np.abs(expm(-1j * float(time) * h)) ** 2
        target = float(np.sum(quantum * quantum))
        if target > n + tolerance or target < components - tolerance:
            raise ValueError("quantum spread is outside the reachable Markov range")
        target = min(float(n), max(float(components), target))
        if target >= n - tolerance:
            solved = 0.0
            classical = np.eye(n)
        elif target <= components + tolerance:
            solved = float("inf")
            classical = stationary.copy()
        else:
            if decay_rates.size == 0 or np.any(decay_rates <= 0):
                raise ValueError("Markov generator has unresolved nonzero decay modes")
            excess = target - components

            def objective(value: float) -> float:
                return float(np.exp(-value * decay_rates).sum() - excess)

            upper = max(1.0, float(time) ** 2)
            for _ in range(1024):
                if objective(upper) <= 0:
                    break
                upper *= 2.0
                if not np.isfinite(upper):
                    raise ValueError("spread matching requires unrepresentably large Markov time")
            else:
                raise ValueError("unable to bracket the spread-matched Markov time")
            solved = float(brentq(objective, 0.0, upper, xtol=1e-13,
                                  rtol=1e-13, maxiter=500))
            classical = expm(solved * generator)
        achieved = float(np.sum(classical * classical))
        if abs(achieved - target) > tolerance:
            raise ValueError("spread-matched Markov kernel failed its trace check")
        matrices.append(classical)
        markov_times.append(solved)
        target_traces.append(target)
        achieved_traces.append(achieved)
    result = _transport_result(matrices, t, w)
    result.update({
        "markov_times": np.asarray(markov_times),
        "generator": generator,
        "target_traces": np.asarray(target_traces),
        "achieved_traces": np.asarray(achieved_traces),
        "connected_components": components,
    })
    return result
