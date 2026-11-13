"""Candidate-conditioned coherent and incoherent group responses.

For a candidate ``e``, the coherent initial state has equal real phase on all
members, while the incoherent reference is an equal mixture of member basis
states.  Their difference is a signed interference statistic, not a hyperedge
probability or evidence of an irreducible many-body interaction.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy import sparse

from .quantum_action import checked_hamiltonian, quantum_action


def canonical_candidate(candidate: Iterable[int], n_nodes: int) -> tuple[int, ...]:
    """Validate one size-at-least-two candidate and return sorted node IDs."""
    raw = list(candidate)
    if any(not isinstance(value, (int, np.integer)) for value in raw):
        raise ValueError("candidate node IDs must be integers")
    nodes = tuple(sorted(int(value) for value in raw))
    if len(nodes) < 2 or len(nodes) > n_nodes:
        raise ValueError("candidate size must be between 2 and N")
    if len(set(nodes)) != len(nodes):
        raise ValueError("candidate contains duplicate node IDs")
    if nodes[0] < 0 or nodes[-1] >= n_nodes:
        raise ValueError("candidate node ID is outside the graph")
    return nodes


def short_time_coefficient(
    hamiltonian: sparse.spmatrix | np.ndarray,
    candidate: Iterable[int],
) -> float:
    """Return the exact coefficient of ``t**2`` in the real-symmetric C(t).

    This is ``-(2/k) sum_{i<j}<H[outside,i], H[outside,j]>``.  It is a
    classical structural control feature, not a quantum-specific signal.
    """
    h = checked_hamiltonian(hamiltonian)
    nodes = canonical_candidate(candidate, h.shape[0])
    outside_mask = np.ones(h.shape[0], dtype=bool)
    outside_mask[list(nodes)] = False
    block = h[outside_mask, :][:, list(nodes)]
    gram = (block.T @ block).toarray()
    return float(-(np.sum(gram) - np.trace(gram)) / len(nodes))


def _initial_states(
    n_nodes: int,
    nodes: tuple[int, ...],
    *,
    exact_incoherent: bool,
    n_phase_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    k = len(nodes)
    references = k if exact_incoherent else n_phase_samples
    states = np.zeros((n_nodes, 1 + references), dtype=np.float64)
    states[list(nodes), 0] = 1.0 / np.sqrt(k)
    if exact_incoherent:
        states[np.asarray(nodes), 1 + np.arange(k)] = 1.0
    else:
        states[np.asarray(nodes), 1:] = (
            rng.integers(0, 2, size=(k, n_phase_samples)) * 2 - 1
        ) / np.sqrt(k)
    return states


def extract_group_quantum_features(
    hamiltonian: sparse.spmatrix | np.ndarray,
    candidates: Sequence[Iterable[int]],
    times: Sequence[float] | np.ndarray,
    *,
    exact_incoherent: bool = True,
    n_phase_samples: int = 16,
    seed: int = 0,
    max_state_columns: int = 64,
) -> tuple[np.ndarray, dict[str, np.ndarray | int | str]]:
    """Extract fixed-time ``[R_coh, R_inc, C]`` for each candidate.

    The output feature matrix has shape ``(M, 3 * len(times))`` with time as
    the outer axis.  Only one bounded candidate-state block and its evolved
    block are resident at once.  Exact incoherent references use one basis
    state per candidate member.  Rademacher references return both per-phase
    sample variance and variance of the estimated mean in diagnostics.

    The ``seed`` and candidate order determine random phases.  For clean
    comparisons, callers should use the same canonical candidate order across
    all method variants.
    """
    h = checked_hamiltonian(hamiltonian)
    grid = np.asarray(times, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all():
        raise ValueError("times must be a nonempty finite one-dimensional sequence")
    if max_state_columns < 1:
        raise ValueError("max_state_columns must be positive")
    if not exact_incoherent and n_phase_samples < 1:
        raise ValueError("n_phase_samples must be positive")
    normalized = [canonical_candidate(candidate, h.shape[0]) for candidate in candidates]
    count = len(normalized)
    r_coh = np.zeros((count, grid.size), dtype=np.float64)
    r_inc = np.zeros_like(r_coh)
    phase_variance = np.zeros_like(r_coh)
    estimator_variance = np.zeros_like(r_coh)
    rng = np.random.default_rng(seed)

    pending: list[tuple[int, tuple[int, ...], np.ndarray]] = []
    pending_columns = 0

    def evaluate_pending() -> None:
        if not pending:
            return
        block = np.concatenate([item[2] for item in pending], axis=1)
        offsets = np.cumsum([0] + [item[2].shape[1] for item in pending])
        for time_index, (_, _, evolved) in enumerate(
            quantum_action(h, block, grid, batch_size=max_state_columns)
        ):
            for local_index, (candidate_index, nodes, _) in enumerate(pending):
                start, stop = offsets[local_index : local_index + 2]
                inside = evolved[np.asarray(nodes), int(start) : int(stop)]
                probabilities = np.sum(np.abs(inside) ** 2, axis=0).real
                r_coh[candidate_index, time_index] = float(probabilities[0])
                reference = probabilities[1:]
                r_inc[candidate_index, time_index] = float(np.mean(reference))
                if not exact_incoherent:
                    if reference.size >= 2:
                        sample_variance = float(np.var(reference, ddof=1))
                        phase_variance[candidate_index, time_index] = sample_variance
                        estimator_variance[candidate_index, time_index] = (
                            sample_variance / reference.size
                        )
                    else:
                        phase_variance[candidate_index, time_index] = np.nan
                        estimator_variance[candidate_index, time_index] = np.nan

    for index, nodes in enumerate(normalized):
        states = _initial_states(
            h.shape[0],
            nodes,
            exact_incoherent=exact_incoherent,
            n_phase_samples=n_phase_samples,
            rng=rng,
        )
        required = states.shape[1]
        if required > max_state_columns:
            raise ValueError(
                "max_state_columns must fit one candidate's coherent and reference states"
            )
        if pending and pending_columns + required > max_state_columns:
            evaluate_pending()
            pending = []
            pending_columns = 0
        pending.append((index, nodes, states))
        pending_columns += required
    evaluate_pending()

    contrast = r_coh - r_inc
    features = np.stack((r_coh, r_inc, contrast), axis=2).reshape(count, -1)
    diagnostics: dict[str, np.ndarray | int | str] = {
        "times": grid.copy(),
        "R_coh": r_coh,
        "R_inc": r_inc,
        "C": contrast,
        "phase_sample_variance": phase_variance,
        "phase_estimator_variance": estimator_variance,
        "incoherent_reference": "exact" if exact_incoherent else "rademacher",
        "n_phase_samples": 0 if exact_incoherent else n_phase_samples,
    }
    return features, diagnostics


def coherent_group_response(
    hamiltonian: sparse.spmatrix | np.ndarray,
    candidate: Iterable[int],
    times: Sequence[float] | np.ndarray,
    *,
    exact_incoherent: bool = True,
    n_phase_samples: int = 16,
    seed: int = 0,
) -> dict[str, np.ndarray | int | str]:
    """Return all response curves and reference diagnostics for one group."""
    nodes = tuple(candidate)
    features, diagnostics = extract_group_quantum_features(
        hamiltonian,
        [nodes],
        times,
        exact_incoherent=exact_incoherent,
        n_phase_samples=n_phase_samples,
        seed=seed,
        max_state_columns=max(64, 1 + n_phase_samples, 1 + len(nodes)),
    )
    return {
        key: value[0] if isinstance(value, np.ndarray) and key != "times" else value
        for key, value in diagnostics.items()
    } | {"features": features[0]}
