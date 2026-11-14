"""Truth-blind progressive clique proposals for the revised P1 protocol.

The returned candidate budget and the number of evaluated search states are
different quantities. This module receives only an observed adjacency matrix.
An optional scoring callback must be fitted on training data before use. No
argument accepts test hyperedge truth.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import isfinite
from typing import Callable, Iterable

import numpy as np
from scipy import sparse

from .graph_io import Edge

ScoreFunction = Callable[[Edge], float]


def _neighbors(A: sparse.spmatrix) -> list[set[int]]:
    graph = sparse.csr_matrix(A, dtype=float)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("A must be square")
    if graph.diagonal().any() or (graph != graph.T).nnz:
        raise ValueError("A must be symmetric with a zero diagonal")
    if graph.data.size and (
        not np.all(np.isfinite(graph.data)) or not np.all(graph.data == 1)
    ):
        raise ValueError("exact projection requires binary adjacency")
    graph.eliminate_zeros()
    return [
        set(int(v) for v in graph.indices[graph.indptr[u] : graph.indptr[u + 1]])
        for u in range(graph.shape[0])
    ]


def _default_score(
    edge: Edge, neighbors: list[set[int]], total_volume: int
) -> float:
    """One minus conductance, used only as an untuned topology heuristic."""
    volume = sum(len(neighbors[u]) for u in edge)
    boundary = volume - len(edge) * (len(edge) - 1)
    denominator = min(volume, total_volume - volume)
    if denominator <= 0:
        return 0.0
    return float(1.0 - boundary / denominator)


def _checked_score(score_fn: ScoreFunction, edge: Edge) -> float:
    value = float(score_fn(edge))
    if not isfinite(value):
        raise ValueError("candidate score must be finite")
    return value


def _select_frontier(
    pool: dict[Edge, float],
    *,
    width: int,
    exploration_fraction: float,
    expand_score_fn: ScoreFunction,
    tie_keys: dict[Edge, float],
    rng: np.random.Generator,
) -> list[Edge]:
    if not pool:
        return []
    ranked = sorted(
        pool,
        key=lambda edge: (-_checked_score(expand_score_fn, edge), tie_keys[edge]),
    )
    if len(ranked) <= width:
        return ranked
    exploratory = min(width - 1, int(round(width * exploration_fraction)))
    if exploratory == 0:
        return ranked[:width]
    leading = ranked[: width - exploratory]
    remainder = ranked[width - exploratory :]
    sampled = rng.choice(len(remainder), size=exploratory, replace=False)
    return leading + [remainder[int(index)] for index in sampled]


def _round_robin(
    ranked: dict[int, list[Edge]],
    sizes: list[int],
    budget: int,
    quotas: dict[int, int] | None,
) -> list[Edge]:
    selected: list[Edge] = []
    cursor = {k: 0 for k in sizes}
    while len(selected) < budget:
        progressed = False
        for k in sizes:
            if quotas is not None and cursor[k] >= quotas[k]:
                continue
            if cursor[k] >= len(ranked[k]):
                continue
            selected.append(ranked[k][cursor[k]])
            cursor[k] += 1
            progressed = True
            if len(selected) >= budget:
                break
        if not progressed:
            break
    return selected


def propose_progressive_candidates(
    A: sparse.spmatrix,
    budget: int,
    *,
    min_k: int = 2,
    max_k: int = 5,
    seed: int = 42,
    per_size_budget: dict[int, int] | None = None,
    score_fn: ScoreFunction | None = None,
    expand_score_fn: ScoreFunction | None = None,
    beam_width: int = 128,
    exploration_fraction: float = 0.2,
    max_visited: int = 200_000,
) -> tuple[list[Edge], dict]:
    """Return distinct clique candidates and a search audit.

    Size-two and size-three cliques are exactly enumerated if their counts do
    not exceed max_visited. Larger cliques are obtained by expanding a bounded
    frontier of immediate subcliques. max_visited caps scored states per size;
    any truncation is reported. The callbacks accept a candidate tuple and
    must close over observation-legal information. If expand_score_fn is
    omitted, score_fn is reused; this is a baseline, not an equivalence claim.
    Explicit per-size budgets are hard caps. Without them, a round-robin
    allocation is an untuned baseline.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    neighbors = _neighbors(A)
    n = len(neighbors)
    if not (2 <= min_k <= max_k <= n):
        raise ValueError("require 2 <= min_k <= max_k <= number of nodes")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (
        isinstance(beam_width, bool)
        or not isinstance(beam_width, int)
        or isinstance(max_visited, bool)
        or not isinstance(max_visited, int)
        or beam_width < 1
        or max_visited < 1
    ):
        raise ValueError("beam_width and max_visited must be positive integers")
    if not 0 <= exploration_fraction < 1:
        raise ValueError("exploration_fraction must be in [0,1)")
    output_sizes = list(range(min_k, max_k + 1))
    if per_size_budget is not None:
        if set(per_size_budget) - set(output_sizes):
            raise ValueError("per_size_budget contains an out-of-range size")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in per_size_budget.values()
        ):
            raise ValueError("per_size_budget values must be nonnegative integers")
        quotas = {k: per_size_budget.get(k, 0) for k in output_sizes}
        if sum(quotas.values()) > budget:
            raise ValueError("per_size_budget sum exceeds budget")
    else:
        quotas = None

    if budget == 0:
        return [], {
            "budget": 0,
            "returned_count": 0,
            "visited_count": 0,
            "per_size_visited": {str(k): 0 for k in range(2, max_k + 1)},
            "per_size_returned": {str(k): 0 for k in output_sizes},
            "per_size_budget": (
                None if quotas is None else {str(k): quotas[k] for k in output_sizes}
            ),
            "per_size_truncated": {
                str(k): False for k in range(2, max_k + 1)
            },
            "generation_counts": {},
            "beam_width": beam_width,
            "exploration_fraction": exploration_fraction,
            "max_visited_per_size": max_visited,
            "seed": seed,
            "score_policy": "topology_heuristic" if score_fn is None else "callback",
            "expand_policy": "same_as_score" if expand_score_fn is None else "callback",
        }

    rng = np.random.default_rng(seed)
    total_volume = sum(len(row) for row in neighbors)
    scorer: ScoreFunction = score_fn or (
        lambda edge: _default_score(edge, neighbors, total_volume)
    )
    expander: ScoreFunction = expand_score_fn or scorer
    pools: dict[int, dict[Edge, float]] = {k: {} for k in range(2, max_k + 1)}
    tie_keys: dict[Edge, float] = {}
    truncated = {k: False for k in pools}
    generation = defaultdict(int)

    def visit(edge: Edge, source: str) -> bool:
        k = len(edge)
        if edge in pools[k]:
            return True
        if len(pools[k]) >= max_visited:
            truncated[k] = True
            return False
        pools[k][edge] = _checked_score(scorer, edge)
        tie_keys[edge] = float(rng.random())
        generation[source] += 1
        return True

    # Randomize enumeration order before applying a cap, avoiding a fixed
    # low-node-ID prefix on dense graphs.
    ordinary_edges = [(u, v) for u in range(n) for v in neighbors[u] if u < v]
    rng.shuffle(ordinary_edges)
    for edge in ordinary_edges:
        if not visit(edge, "ordinary_edge"):
            break

    if max_k >= 3:
        for u, v in ordinary_edges:
            common = [w for w in neighbors[u] & neighbors[v] if w > v]
            rng.shuffle(common)
            for w in common:
                if not visit((u, v, w), "triangle"):
                    break
            if truncated[3]:
                break

    # A bounded beam may discard a true edge's prefix. Exploration reduces
    # that risk without claiming a recall guarantee.
    for k in range(4, max_k + 1):
        frontier = _select_frontier(
            pools[k - 1],
            width=beam_width,
            exploration_fraction=exploration_fraction,
            expand_score_fn=expander,
            tie_keys=tie_keys,
            rng=rng,
        )
        for parent in frontier:
            common = set.intersection(*(neighbors[u] for u in parent))
            common.difference_update(parent)
            available = list(common)
            rng.shuffle(available)
            for u in available:
                child = tuple(sorted((*parent, int(u))))
                if not visit(child, "beam_expansion"):
                    break
            if truncated[k]:
                break

    ranked = {
        k: sorted(pools[k], key=lambda edge: (-pools[k][edge], tie_keys[edge]))
        for k in output_sizes
    }
    selected = _round_robin(ranked, output_sizes, budget, quotas)
    audit = {
        "budget": budget,
        "returned_count": len(selected),
        "visited_count": sum(len(pool) for pool in pools.values()),
        "per_size_visited": {str(k): len(pools[k]) for k in pools},
        "per_size_returned": {
            str(k): sum(len(edge) == k for edge in selected) for k in output_sizes
        },
        "per_size_budget": (
            None if quotas is None else {str(k): quotas[k] for k in output_sizes}
        ),
        "per_size_truncated": {str(k): truncated[k] for k in pools},
        "generation_counts": dict(sorted(generation.items())),
        "beam_width": beam_width,
        "exploration_fraction": exploration_fraction,
        "max_visited_per_size": max_visited,
        "seed": seed,
        "score_policy": "topology_heuristic" if score_fn is None else "callback",
        "expand_policy": "same_as_score" if expand_score_fn is None else "callback",
    }
    return selected, audit


def is_clique_from_neighbors(
    neighbors: list[set[int]], edge: Iterable[int]
) -> bool:
    """Validation helper for tests and independent result audits."""
    nodes = tuple(edge)
    return len(set(nodes)) == len(nodes) and all(
        v in neighbors[u] for u, v in combinations(nodes, 2)
    )
