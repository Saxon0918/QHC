"""Clique projection of event hyperedges into binary or multiplicity graphs."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable

import numpy as np
from scipy import sparse

from .graph_io import canonicalize_edge


def project_hypergraph(hyperedges: Iterable[Iterable[int]], n_nodes: int,
                       mode: str = "binary") -> sparse.csr_matrix:
    """Project a hyperedge *event list* onto an undirected, loop-free graph.

    In ``binary`` mode, one or more events containing a pair yield weight 1.
    In ``multiplicity`` mode, each event contributes one to every member pair,
    including repeated events with the same member set. The input event list is
    not silently deduplicated. Node IDs are 0-based in ``[0, n_nodes)``.
    """
    if not isinstance(n_nodes, int) or n_nodes < 0:
        raise ValueError("n_nodes must be a nonnegative integer")
    if mode not in {"binary", "multiplicity"}:
        raise ValueError("mode must be 'binary' or 'multiplicity'")
    counts: Counter[tuple[int, int]] = Counter()
    for nodes in hyperedges:
        edge = canonicalize_edge(nodes, n_nodes=n_nodes)
        for pair in combinations(edge, 2):
            counts[pair] += 1
    if not counts:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=np.int64)
    rows: list[int] = []
    cols: list[int] = []
    values: list[int] = []
    for (i, j), count in counts.items():
        value = 1 if mode == "binary" else count
        rows.extend((i, j))
        cols.extend((j, i))
        values.extend((value, value))
    return sparse.csr_matrix((np.asarray(values, dtype=np.int64),
                              (rows, cols)), shape=(n_nodes, n_nodes))


def incidence_matrix(hyperedges: Iterable[Iterable[int]], n_nodes: int,
                     deduplicate: bool = False) -> sparse.csr_matrix:
    """Create an N-by-M binary incidence matrix; duplicate events are optional."""
    if not isinstance(n_nodes, int) or n_nodes < 0:
        raise ValueError("n_nodes must be a nonnegative integer")
    edges = [canonicalize_edge(edge, n_nodes=n_nodes) for edge in hyperedges]
    if deduplicate:
        edges = sorted(set(edges))
    row = [node for edge in edges for node in edge]
    col = [j for j, edge in enumerate(edges) for _ in edge]
    B = sparse.csr_matrix((np.ones(len(row), dtype=np.int8), (row, col)),
                          shape=(n_nodes, len(edges)))
    return B
