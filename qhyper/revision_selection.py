"""Truth-blind structure selection for the revised reconstruction protocol.

Pure weighted coverage is monotone submodular. The two-part code below is a
decodable model-selection baseline, not a guarantee of true hypergraph recovery.
When candidate scores are included the result is a conditional energy because
those scores can already depend on the observed graph.
"""

from __future__ import annotations

from itertools import combinations
from math import ceil, comb, floor, log2, log
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.special import logsumexp

from .graph_io import Edge, canonicalize_edge


Pair = tuple[int, int]


def observed_pairs(A: sparse.spmatrix | np.ndarray) -> set[Pair]:
    """Positive unordered dyads of a symmetric loop-free binary graph."""
    graph = sparse.csr_matrix(A)
    if graph.shape[0] != graph.shape[1] or graph.diagonal().any():
        raise ValueError("A must be square and loop-free")
    if (graph != graph.T).nnz or (graph.data.size and
                                  not np.all(np.isin(graph.data, (0, 1)))):
        raise ValueError("A must be symmetric and binary")
    upper = sparse.triu(graph, k=1).tocoo()
    return {(int(i), int(j)) for i, j in zip(upper.row, upper.col)}


def _unique_cliques(candidates: Iterable[Iterable[int]], pairs: set[Pair],
                    n_nodes: int) -> list[Edge]:
    unique = sorted({canonicalize_edge(edge, n_nodes=n_nodes) for edge in candidates})
    if any(any(pair not in pairs for pair in combinations(edge, 2)) for edge in unique):
        raise ValueError("every candidate must be a clique in exact binary projection")
    return unique


def _pair_sets(candidates: Sequence[Edge]) -> list[set[Pair]]:
    return [set(combinations(edge, 2)) for edge in candidates]


def coverage_value(A: sparse.spmatrix | np.ndarray,
                   selected: Iterable[Iterable[int]],
                   weights: dict[Pair, float] | None = None) -> float:
    """Weighted number of distinct observed dyads covered by selected cliques."""
    pairs = observed_pairs(A)
    if weights is not None and (set(weights) - pairs or
                                any(not np.isfinite(w) or w < 0 for w in weights.values())):
        raise ValueError("weights must be finite nonnegative observed-dyad weights")
    edges = _unique_cliques(selected, pairs, sparse.csr_matrix(A).shape[0])
    covered = set().union(*_pair_sets(edges)) if edges else set()
    return float(sum(1.0 if weights is None else weights.get(pair, 1.0)
                     for pair in covered))


def select_top_m(candidates: Iterable[Iterable[int]], scores: Sequence[float],
                 count: int) -> list[Edge]:
    """Select a validation-fixed number of distinct highest-score candidates."""
    edges = [canonicalize_edge(edge) for edge in candidates]
    values = np.asarray(scores, dtype=float)
    if len(edges) != len(set(edges)):
        raise ValueError("top-M candidates must be unique")
    if values.shape != (len(edges),) or not np.isfinite(values).all():
        raise ValueError("scores must be finite and aligned")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    ranked = sorted(range(len(edges)), key=lambda i: (-values[i], edges[i]))
    return [edges[i] for i in ranked[:count]]


def select_greedy_coverage(A: sparse.spmatrix | np.ndarray,
                           candidates: Iterable[Iterable[int]], *,
                           max_selected: int,
                           weights: dict[Pair, float] | None = None) -> list[Edge]:
    """Greedy maximum coverage; 1-1/e applies only to this pure objective."""
    if isinstance(max_selected, bool) or max_selected < 0:
        raise ValueError("max_selected must be nonnegative")
    pairs = observed_pairs(A)
    if weights is not None and (set(weights) - pairs or
                                any(not np.isfinite(w) or w < 0 for w in weights.values())):
        raise ValueError("weights must be finite nonnegative observed-dyad weights")
    edges = _unique_cliques(candidates, pairs, sparse.csr_matrix(A).shape[0])
    edge_pairs = _pair_sets(edges)
    uncovered = set(pairs)
    remaining = set(range(len(edges)))
    chosen: list[Edge] = []
    while len(chosen) < max_selected and remaining:
        gains = {i: sum(1.0 if weights is None else weights.get(p, 1.0)
                        for p in edge_pairs[i] & uncovered) for i in remaining}
        index = min(remaining, key=lambda i: (-gains[i], edges[i]))
        if gains[index] <= 0:
            break
        chosen.append(edges[index])
        uncovered.difference_update(edge_pairs[index])
        remaining.remove(index)
    return chosen


def gamma_code_length(value: int) -> int:
    """Elias gamma length of a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("gamma input must be a positive integer")
    return 2 * floor(log2(value)) + 1


def _edge_code_length(n_nodes: int, size: int) -> int:
    # Gamma-code k-1, followed by a fixed-width rank of this k-subset.
    return gamma_code_length(size - 1) + ceil(log2(comb(n_nodes, size)))


def _residual_code_length(n_nodes: int, count: int) -> int:
    # Gamma-code count+1, then fixed-width positions of unexplained A dyads.
    if count == 0:
        return gamma_code_length(1)
    return gamma_code_length(count + 1) + count * ceil(log2(comb(n_nodes, 2)))


def two_part_description_length(A: sparse.spmatrix | np.ndarray,
                                selected: Iterable[Iterable[int]]) -> int:
    """Code selected cliques plus unexplained positive dyads.

    Residual dyads are observation errors, not predicted hyperedges. This is
    a valid two-part code but does not solve projection non-identifiability.
    """
    pairs = observed_pairs(A)
    n_nodes = sparse.csr_matrix(A).shape[0]
    edges = _unique_cliques(selected, pairs, n_nodes)
    covered = set().union(*_pair_sets(edges)) if edges else set()
    return (gamma_code_length(len(edges) + 1) +
            sum(_edge_code_length(n_nodes, len(edge)) for edge in edges) +
            _residual_code_length(n_nodes, len(pairs - covered)))


def select_mdl_greedy(A: sparse.spmatrix | np.ndarray,
                      candidates: Iterable[Iterable[int]], *,
                      max_selected: int | None = None,
                      scores: Sequence[float] | None = None,
                      score_weight: float = 0.0) -> list[Edge]:
    """Greedy two-part code with optional conditional score reward.

    Scores align with input candidates. With score_weight>0 this is a
    validation-tuned conditional energy, not MDL or a Bayesian posterior.
    No approximation guarantee applies to this complete objective.
    """
    n_nodes = sparse.csr_matrix(A).shape[0]
    raw_edges = [canonicalize_edge(edge, n_nodes=n_nodes) for edge in candidates]
    pairs = observed_pairs(A)
    edges = _unique_cliques(raw_edges, pairs, n_nodes)
    if max_selected is None:
        max_selected = len(edges)
    if isinstance(max_selected, bool) or max_selected < 0:
        raise ValueError("max_selected must be nonnegative")
    if not np.isfinite(score_weight) or score_weight < 0:
        raise ValueError("score_weight must be finite and nonnegative")
    reward = {edge: 0.0 for edge in edges}
    if scores is not None:
        values = np.asarray(scores, dtype=float)
        if values.shape != (len(raw_edges),) or not np.isfinite(values).all() or np.any(
                (values < 0) | (values > 1)):
            raise ValueError("scores must be aligned probabilities in [0,1]")
        if len(set(raw_edges)) != len(raw_edges):
            raise ValueError("duplicate candidates make score alignment ambiguous")
        for edge, value in zip(raw_edges, values):
            clipped = float(np.clip(value, 1e-12, 1 - 1e-12))
            reward[edge] = log(clipped / (1 - clipped))
    elif score_weight:
        raise ValueError("positive score_weight requires scores")
    edge_pairs = _pair_sets(edges)
    uncovered = set(pairs)
    selected: list[Edge] = []
    remaining = set(range(len(edges)))
    while len(selected) < max_selected and remaining:
        count = len(uncovered)
        count_delta = gamma_code_length(len(selected) + 2) - gamma_code_length(len(selected) + 1)
        best: tuple[float, int] | None = None
        for index in remaining:
            gain = len(edge_pairs[index] & uncovered)
            residual_delta = (_residual_code_length(n_nodes, count - gain)
                              - _residual_code_length(n_nodes, count))
            delta = (count_delta + _edge_code_length(n_nodes, len(edges[index]))
                     + residual_delta - score_weight * reward[edges[index]])
            if best is None or delta < best[0] - 1e-12 or (
                    abs(delta - best[0]) <= 1e-12 and edges[index] < edges[best[1]]):
                best = (delta, index)
        assert best is not None
        if best[0] >= -1e-12:
            break
        index = best[1]
        selected.append(edges[index])
        uncovered.difference_update(edge_pairs[index])
        remaining.remove(index)
    return selected


def _projection_incidence(A: sparse.spmatrix | np.ndarray,
                          candidates: Iterable[Iterable[int]],
                          probabilities: Sequence[float]) -> tuple[list[Edge], np.ndarray, sparse.csc_matrix]:
    """Validate an exact-projection universe and its training-only priors."""
    pairs = sorted(observed_pairs(A))
    n_nodes = sparse.csr_matrix(A).shape[0]
    edges = [canonicalize_edge(edge, n_nodes=n_nodes) for edge in candidates]
    if len(edges) != len(set(edges)):
        raise ValueError("candidate universe contains duplicates")
    _unique_cliques(edges, set(pairs), n_nodes)
    pi = np.asarray(probabilities, dtype=float)
    if pi.shape != (len(edges),) or not np.isfinite(pi).all() or np.any((pi <= 0) | (pi >= 1)):
        raise ValueError("prior probabilities must align and lie strictly in (0,1)")
    pair_index = {pair: i for i, pair in enumerate(pairs)}
    rows: list[int] = []
    cols: list[int] = []
    for column, edge in enumerate(edges):
        for pair in combinations(edge, 2):
            rows.append(pair_index[pair])
            cols.append(column)
    incidence = sparse.csc_matrix((np.ones(len(rows)), (rows, cols)),
                                  shape=(len(pairs), len(edges)), dtype=float)
    if pairs and np.any(np.asarray(incidence.sum(axis=1)).ravel() == 0):
        raise ValueError("candidate universe does not cover every observed dyad; "
                         "add and count explicit fallback dyads or use a noisy model")
    return edges, pi, incidence


def select_exact_projection_map(A: sparse.spmatrix | np.ndarray,
                                candidates: Iterable[Iterable[int]],
                                prior_probabilities: Sequence[float], *,
                                time_limit_seconds: float | None = None) -> list[Edge]:
    """MAP hypergraph under independent training-only Bernoulli priors.

    The observation likelihood is the deterministic clique projection.
    Every observed dyad must have candidate support; fallback dyads, if used,
    are supplied explicitly and count in the caller's candidate universe.
    This MILP may be expensive, and no time-limited nonoptimal solution is
    silently presented as MAP.
    """
    edges, pi, incidence = _projection_incidence(A, candidates, prior_probabilities)
    if not edges:
        return []
    if time_limit_seconds is not None and (
            not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0):
        raise ValueError("time_limit_seconds must be positive and finite")
    costs = np.log1p(-pi) - np.log(pi)
    constraints = (LinearConstraint(incidence, lb=np.ones(incidence.shape[0]),
                                    ub=np.full(incidence.shape[0], np.inf))
                   if incidence.shape[0] else None)
    options = {} if time_limit_seconds is None else {"time_limit": float(time_limit_seconds)}
    result = milp(c=costs, integrality=np.ones(len(edges)),
                  bounds=Bounds(np.zeros(len(edges)), np.ones(len(edges))),
                  constraints=constraints, options=options)
    if result.status != 0 or result.x is None:
        raise RuntimeError(f"exact projection MAP was not proven optimal: {result.message}")
    return [edge for edge, chosen in zip(edges, result.x) if chosen > 0.5]


def exact_posterior_small(A: sparse.spmatrix | np.ndarray,
                          candidates: Iterable[Iterable[int]],
                          prior_probabilities: Sequence[float], *,
                          max_candidates: int = 16) -> dict[str, object]:
    """Enumerate a small exact-projection posterior to expose ambiguity.

    This is a mathematical oracle for tiny synthetic graphs, never a scalable
    reconstruction method. Priors must be learned without current test A.
    """
    edges, pi, incidence = _projection_incidence(A, candidates, prior_probabilities)
    if len(edges) > max_candidates:
        raise ValueError("candidate universe too large for exact posterior enumeration")
    if isinstance(max_candidates, bool) or max_candidates < 0:
        raise ValueError("max_candidates must be nonnegative")
    log_pi = np.log(pi)
    log_not = np.log1p(-pi)
    assignments: list[np.ndarray] = []
    log_weights: list[float] = []
    for mask in range(1 << len(edges)):
        z = np.fromiter(((mask >> i) & 1 for i in range(len(edges))), dtype=int)
        if incidence.shape[0] and np.any(incidence @ z < 1):
            continue
        assignments.append(z)
        log_weights.append(float(np.dot(z, log_pi) + np.dot(1 - z, log_not)))
    log_normalizer = float(logsumexp(log_weights))
    normalized_log_weights = np.asarray(log_weights) - log_normalizer
    weights = np.exp(normalized_log_weights)
    matrix = np.vstack(assignments) if assignments else np.zeros((0, len(edges)))
    marginals = weights @ matrix
    # Some feasible structures have posterior mass below the smallest float.
    # Their contribution tends to zero; avoid the indeterminate 0 * log(0).
    positive = weights > 0
    entropy = float(-np.dot(weights[positive], normalized_log_weights[positive]))
    return {"candidates": edges,
            "marginal_probability": marginals.tolist(),
            "posterior_entropy_nats": entropy,
            "feasible_structure_count": len(assignments),
            "map_structure": [edges[i] for i, value in enumerate(
                assignments[int(np.argmax(log_weights))]) if value]}
