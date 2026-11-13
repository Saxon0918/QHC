"""Truth-blind, budgeted clique proposals and their separate oracle coverage.

No function in the proposal path accepts ground-truth hyperedges. The coverage
functions are evaluation-only and must be called after a candidate pool is fixed.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import comb
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from scipy import sparse

from .graph_io import CandidateRecord, Edge, canonicalize_edge, canonicalize_edges


def _adjacency_sets(A: sparse.spmatrix) -> list[set[int]]:
    matrix = sparse.csr_matrix(A)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be square")
    if matrix.diagonal().any():
        raise ValueError("adjacency must have zero diagonal")
    if (matrix != matrix.T).nnz:
        raise ValueError("adjacency must be symmetric")
    matrix = matrix.copy()
    matrix.eliminate_zeros()
    return [set(int(i) for i in matrix.indices[matrix.indptr[v]:matrix.indptr[v + 1]])
            for v in range(matrix.shape[0])]


def is_clique(A: sparse.spmatrix, nodes: Iterable[int]) -> bool:
    """Check the necessary true-edge condition for exact binary projection."""
    neighbors = _adjacency_sets(A)
    edge = canonicalize_edge(nodes, n_nodes=len(neighbors))
    return all(j in neighbors[i] for i, j in combinations(edge, 2))


def _bounded_maximal_cliques(neighbors: list[set[int]],
                             search_budget: int,
                             result_budget: int) -> list[Edge]:
    """Bron--Kerbosch with hard call/result caps, deterministic tie breaking."""
    results: list[Edge] = []
    calls = 0

    def visit(r: set[int], p: set[int], x: set[int]) -> None:
        nonlocal calls
        if calls >= search_budget or len(results) >= result_budget:
            return
        calls += 1
        if not p and not x:
            results.append(tuple(sorted(r)))
            return
        union = p | x
        pivot = max(union, key=lambda u: (len(p & neighbors[u]), -u)) if union else None
        branch = sorted(p - (neighbors[pivot] if pivot is not None else set()))
        for v in branch:
            if calls >= search_budget or len(results) >= result_budget:
                break
            visit(r | {v}, p & neighbors[v], x & neighbors[v])
            p.remove(v)
            x.add(v)

    visit(set(), set(range(len(neighbors))), set())
    return results


def _subsets_of_clique(clique: Edge, k: int, cap: int,
                       rng: np.random.Generator) -> list[Edge]:
    if len(clique) < k or cap <= 0:
        return []
    total = comb(len(clique), k)
    if total <= cap:
        return list(combinations(clique, k))
    subsets: set[Edge] = set()
    attempts = 0
    while len(subsets) < cap and attempts < max(100, 20 * cap):
        attempts += 1
        drawn = tuple(sorted(int(i) for i in rng.choice(clique, size=k, replace=False)))
        subsets.add(drawn)
    return sorted(subsets)


def propose_candidates(A: sparse.spmatrix, X: Optional[np.ndarray] = None,
                       budget: int = 5000, min_k: int = 3, max_k: int = 5,
                       seed: int = 42,
                       per_size_budget: Optional[dict[int, int]] = None,
                       generators: Optional[Sequence[str]] = None,
                       graph_id: str = "graph",
                       return_records: bool = False) -> list[Any]:
    """Propose unique cliques under a fixed, truth-independent total budget.

    The selected channels are bounded maximal-clique/subclique sampling and/or
    local expansion from ordinary edges. Candidate order is reproducible for a
    fixed input and seed, and is the order used for Recall@Budget. ``X`` is
    reserved for a future separately reported feature-driven protocol; it is
    not used in exact-projection P1. Pass ``min_k=2`` when size-2 ground truth
    must be covered. The example research config begins at 3 and therefore
    necessarily misses every true size-2 hyperedge.
    """
    if not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    neighbors = _adjacency_sets(A)
    n = len(neighbors)
    if min_k < 2 or max_k < min_k or max_k > n:
        raise ValueError("require 2 <= min_k <= max_k <= n_nodes")
    if per_size_budget is not None:
        if any(k < min_k or k > max_k or value < 0
               for k, value in per_size_budget.items()):
            raise ValueError("per_size_budget contains an invalid size or count")
    allowed_generators = {"local_clique", "maximal_clique_subsets"}
    if generators is None:
        selected_generators = allowed_generators
    else:
        if isinstance(generators, str):
            raise ValueError("generators must be a sequence, not one string")
        requested = tuple(generators)
        if (not requested or len(set(requested)) != len(requested)
                or set(requested) - allowed_generators):
            raise ValueError("generators must be unique nonempty names from the supported channels")
        selected_generators = set(requested)
    if budget == 0:
        return []
    rng = np.random.default_rng(seed)
    source_by_edge: dict[Edge, set[str]] = defaultdict(set)
    sizes = list(range(min_k, max_k + 1))
    proposal_cap_per_size = max(32, 4 * budget // len(sizes))
    proposal_counts = defaultdict(int)

    def add(edge: Edge, source: str) -> None:
        k = len(edge)
        if k < min_k or k > max_k:
            return
        if edge not in source_by_edge and proposal_counts[k] >= proposal_cap_per_size:
            return
        if edge not in source_by_edge:
            proposal_counts[k] += 1
        source_by_edge[edge].add(source)

    if "maximal_clique_subsets" in selected_generators:
        maximal = _bounded_maximal_cliques(
            neighbors,
            search_budget=max(1000, 40 * budget),
            result_budget=max(100, 2 * budget),
        )
        maximal = [clique for clique in maximal if len(clique) >= min_k]
        if maximal:
            # Cycle cliques so one enormous component cannot exhaust all proposals.
            ordering = rng.permutation(len(maximal))
            for idx in ordering:
                clique = maximal[int(idx)]
                for k in sizes:
                    remaining = proposal_cap_per_size - proposal_counts[k]
                    if remaining <= 0:
                        continue
                    for edge in _subsets_of_clique(clique, k, min(remaining, 8), rng):
                        add(edge, "maximal_clique")

    if "local_clique" in selected_generators:
        ordinary_edges = sorted((i, j) for i in range(n) for j in neighbors[i] if i < j)
        if ordinary_edges:
            seed_order = rng.permutation(len(ordinary_edges))[:min(len(ordinary_edges),
                                                                     max(100, 8 * budget))]
            for edge_idx in seed_order:
                edge = ordinary_edges[int(edge_idx)]
                if min_k <= 2:
                    add(edge, "local_clique")
                grown = list(edge)
                common = neighbors[edge[0]] & neighbors[edge[1]]
                while common and len(grown) < max_k:
                    available = sorted(common - set(grown))
                    if not available:
                        break
                    chosen = int(rng.choice(available))
                    grown.append(chosen)
                    common &= neighbors[chosen]
                    add(tuple(sorted(grown)), "local_clique")
                if all(proposal_counts[k] >= proposal_cap_per_size for k in sizes):
                    break

    by_size: dict[int, list[Edge]] = {}
    for k in sizes:
        entries = sorted(edge for edge in source_by_edge if len(edge) == k)
        if entries:
            order = rng.permutation(len(entries))
            by_size[k] = [entries[int(i)] for i in order]
        else:
            by_size[k] = []

    # Equal initial size quotas unless explicit hard per-size caps are given.
    # With implicit quotas, unused slots are redistributed round-robin.
    equal, extra = divmod(budget, len(sizes))
    quotas = {k: equal + int(i < extra) for i, k in enumerate(sizes)}
    if per_size_budget is not None:
        quotas = {k: per_size_budget.get(k, 0) for k in sizes}
    selected: list[Edge] = []
    used: set[Edge] = set()
    cursor = {k: 0 for k in sizes}
    for k in sizes:
        for edge in by_size[k][:quotas[k]]:
            if len(selected) >= budget:
                break
            selected.append(edge)
            used.add(edge)
            cursor[k] += 1
    while per_size_budget is None and len(selected) < budget:
        progressed = False
        for k in sizes:
            while cursor[k] < len(by_size[k]) and by_size[k][cursor[k]] in used:
                cursor[k] += 1
            if cursor[k] < len(by_size[k]):
                edge = by_size[k][cursor[k]]
                cursor[k] += 1
                selected.append(edge)
                used.add(edge)
                progressed = True
                if len(selected) >= budget:
                    break
        if not progressed:
            break
    if return_records:
        return [CandidateRecord(graph_id=graph_id, node_ids=edge,
                                proposal_sources=tuple(sorted(source_by_edge[edge])))
                for edge in selected]
    return selected


def candidate_coverage(candidates: Iterable[Iterable[int]],
                       hyperedges_truth: Iterable[Iterable[int]]) -> dict[str, Any]:
    """Evaluation-only oracle coverage of exact, deduplicated member sets."""
    candidate_set = set(canonicalize_edges(
        (candidate.node_ids if isinstance(candidate, CandidateRecord) else candidate
         for candidate in candidates), deduplicate=True))
    truth_set = set(canonicalize_edges(hyperedges_truth, deduplicate=True))
    covered = candidate_set & truth_set
    by_size: dict[int, dict[str, Any]] = {}
    for k in sorted({len(e) for e in truth_set}):
        truth_k = {e for e in truth_set if len(e) == k}
        covered_k = truth_k & candidate_set
        by_size[k] = {
            "truth_count": len(truth_k),
            "covered_count": len(covered_k),
            "recall": len(covered_k) / len(truth_k),
        }
    return {
        "candidate_count": len(candidate_set),
        "truth_count": len(truth_set),
        "covered_count": len(covered),
        "recall": len(covered) / len(truth_set) if truth_set else None,
        "by_size": by_size,
    }


def recall_at_budget(ordered_candidates: Sequence[Iterable[int]],
                     hyperedges_truth: Iterable[Iterable[int]],
                     budgets: Iterable[int]) -> list[dict[str, Any]]:
    """Evaluate the fixed proposal order at prespecified candidate budgets."""
    truth = list(hyperedges_truth)
    normalized = [canonicalize_edge(edge.node_ids if isinstance(edge, CandidateRecord)
                                    else edge) for edge in ordered_candidates]
    results = []
    for budget in budgets:
        if not isinstance(budget, int) or budget < 0:
            raise ValueError("budgets must be nonnegative integers")
        entry = candidate_coverage(normalized[:budget], truth)
        entry["budget"] = int(budget)
        results.append(entry)
    return results
