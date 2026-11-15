"""Label-free local proposals for *latent* hypergraph construction.

Every proposed group is contained in the closed neighborhood of at least one
anchor vertex in an explicitly chosen support graph. By default that graph is
the observed topology alone; optionally it is the union of observed edges and
mutual-kNN positive-cosine feature edges. In particular, two neighbors of an
anchor need not be adjacent: the output intentionally includes noncliques.
This is appropriate for latent construction, but it is not a candidate
generator for exact noiseless hyperedge reconstruction.

The search is bounded: each anchor considers only its highest-ranked local
neighbors, and each cardinality retains its strongest proposals before the
final round-robin budget allocation. Ranking uses the chosen support graph,
never node labels or true hyperedges.
Lexicographic ties make runs reproducible, while a binding budget can break
permutation equivariance on graphs with topological ties.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
from scipy import sparse

from .hamiltonian import checked_adjacency, feature_similarity_adjacency


Group = tuple[int, ...]


def propose_groups(
    adjacency: sparse.spmatrix | np.ndarray,
    *,
    features: np.ndarray | None = None,
    feature_k: int = 10,
    min_size: int = 2,
    max_size: int = 5,
    max_candidates: int = 5000,
    include_pairs: bool = True,
    neighborhood_cap: int = 8,
) -> tuple[Group, ...]:
    """Return a deterministic, budgeted pool of local groups.

    A size-``k`` group with ``k >= 3`` is made from an anchor plus ``k-1``
    members of its open neighborhood in the support graph. When ``features``
    is supplied, this graph is the union of observed edges and the mutual-kNN
    positive-cosine feature graph; the optional feature protocol must be
    reported separately from topology-only construction. Feature values do
    not directly enter the ranking once the support graph is built.

    The anchor's neighbors are ranked by
    common-neighbor count, then by degree similarity, then by node index;
    at most ``neighborhood_cap`` are considered. Support-graph edges supply
    size-two candidates when ``include_pairs`` and ``min_size <= 2``.

    Groups are ranked *within size* by the number of closed neighborhoods
    containing all members, then internal graph density. Final allocation is
    round-robin over sizes, so ordinary pairs do not automatically exhaust
    the budget. Returned groups are sorted and unique. Neither labels nor
    true hyperedges are accepted or inspected.
    """
    for name, value in (
        ("feature_k", feature_k),
        ("min_size", min_size), ("max_size", max_size),
        ("max_candidates", max_candidates), ("neighborhood_cap", neighborhood_cap),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if min_size < 2 or max_size < min_size:
        raise ValueError("require 2 <= min_size <= max_size")
    if max_candidates < 0 or neighborhood_cap < 1 or feature_k < 1:
        raise ValueError("max_candidates must be nonnegative; neighborhood_cap and feature_k positive")
    if not isinstance(include_pairs, bool):
        raise ValueError("include_pairs must be boolean")

    graph = checked_adjacency(adjacency)
    n = graph.shape[0]
    if features is not None:
        x = np.asarray(features)
        if x.ndim != 2 or x.shape[0] != n:
            raise ValueError("features must be an N-by-d matrix aligned with adjacency")
        feature_graph = feature_similarity_adjacency(x, k=feature_k)
        graph = checked_adjacency(graph.maximum(feature_graph))
    sizes = [k for k in range(min_size, min(max_size, n) + 1)
             if include_pairs or k != 2]
    if max_candidates == 0 or not sizes:
        return ()

    neighbors = [
        set(int(v) for v in graph.indices[graph.indptr[u]:graph.indptr[u + 1]])
        for u in range(n)
    ]
    closed = [row | {u} for u, row in enumerate(neighbors)]
    degrees = [len(row) for row in neighbors]

    def rank(group: Group) -> tuple[float, float, Group]:
        common = set.intersection(*(closed[u] for u in group))
        inside_pairs = sum(v in neighbors[u] for u, v in combinations(group, 2))
        density = inside_pairs / comb(len(group), 2)
        return (-float(len(common)), -float(density), group)

    # A bounded cache per cardinality avoids materializing the combinatorial
    # union of all closed neighborhoods on a large graph. Each candidate has
    # an intrinsic score, so pruning does not depend on proposal multiplicity.
    pools: dict[int, dict[Group, tuple[float, float, Group]]] = {
        k: {} for k in sizes
    }
    cap_per_size = max_candidates

    def offer(group: Group) -> None:
        k = len(group)
        pool = pools[k]
        if group in pool:
            return
        pool[group] = rank(group)
        if len(pool) > 2 * cap_per_size:
            best = sorted(pool, key=pool.__getitem__)[:cap_per_size]
            pools[k] = {edge: pool[edge] for edge in best}

    if 2 in pools:
        for u in range(n):
            for v in neighbors[u]:
                if u < v:
                    offer((u, v))

    for anchor in range(n):
        adjacent = sorted(
            neighbors[anchor],
            key=lambda v: (
                -len(neighbors[anchor] & neighbors[v]),
                abs(degrees[anchor] - degrees[v]),
                v,
            ),
        )[:neighborhood_cap]
        for k in sizes:
            if k < 3 or len(adjacent) < k - 1:
                continue
            for members in combinations(adjacent, k - 1):
                offer(tuple(sorted((anchor, *members))))

    ranked = {
        k: sorted(pool, key=pool.__getitem__)[:cap_per_size]
        for k, pool in pools.items()
    }
    cursor = {k: 0 for k in sizes}
    selected: list[Group] = []
    while len(selected) < max_candidates:
        progressed = False
        for k in sizes:
            if cursor[k] >= len(ranked[k]):
                continue
            selected.append(ranked[k][cursor[k]])
            cursor[k] += 1
            progressed = True
            if len(selected) == max_candidates:
                break
        if not progressed:
            break
    return tuple(sorted(selected))
