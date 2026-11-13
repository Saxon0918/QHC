"""Reproducible synthetic hypergraphs and the triangle ambiguity witness.

These generators preserve event truth and construct the observation only from
that truth. They make no claim that arbitrary projected hypergraphs are
recoverable. S1 includes pair-event decoys; S2 deliberately increases overlap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
from scipy import sparse

from .graph_io import Edge, GraphRecord
from .projection import project_hypergraph


@dataclass(frozen=True)
class SyntheticConfig:
    """Inputs known before drawing a synthetic train/validation/test graph."""

    case: str = "S1"
    n_nodes: int = 60
    num_hyperedges: int = 40
    edge_sizes: tuple[int, ...] = (3, 4)
    n_blocks: int = 4
    feature_signal: float = 0.0
    edge_add_prob: float = 0.0
    edge_drop_prob: float = 0.0
    include_multiplicity: bool = False

    def __post_init__(self) -> None:
        if self.case not in {"S1", "S2"}:
            raise ValueError("case must be S1 or S2; use triangle_ambiguity_example for S3")
        if self.n_nodes < 4 or self.num_hyperedges < 1:
            raise ValueError("n_nodes >= 4 and num_hyperedges >= 1 are required")
        if not self.edge_sizes or any(k < 2 or k > self.n_nodes for k in self.edge_sizes):
            raise ValueError("edge_sizes must lie in [2, n_nodes]")
        if self.n_blocks < 1 or self.n_blocks > self.n_nodes:
            raise ValueError("n_blocks must lie in [1, n_nodes]")
        if self.feature_signal < 0:
            raise ValueError("feature_signal must be nonnegative")
        if not 0 <= self.edge_add_prob < 1 or not 0 <= self.edge_drop_prob < 1:
            raise ValueError("noise probabilities must lie in [0, 1)")


@dataclass
class SyntheticSample:
    """One generated graph, optional multiplicity view, and exact parameters."""

    graph: GraphRecord
    multiplicity_csr: Optional[sparse.csr_matrix]
    parameters: dict[str, Any]
    seed: int


def triangle_ambiguity_example() -> tuple[GraphRecord, GraphRecord]:
    """Two distinct truths with exactly the same K3 binary observation."""
    one_triad: list[Edge] = [(0, 1, 2)]
    three_pairs: list[Edge] = [(0, 1), (0, 2), (1, 2)]
    A = project_hypergraph(one_triad, 3, mode="binary")
    B = project_hypergraph(three_pairs, 3, mode="binary")
    assert (A != B).nnz == 0
    first = GraphRecord(
        graph_id="triangle-one-hyperedge", num_nodes=3, adjacency_csr=A,
        hyperedges_truth=one_triad,
        provenance={"dataset": "synthetic-S3", "world": "one-triad"},
    )
    second = GraphRecord(
        graph_id="triangle-three-pairs", num_nodes=3, adjacency_csr=B,
        hyperedges_truth=three_pairs,
        provenance={"dataset": "synthetic-S3", "world": "three-pairs"},
    )
    return first, second


def _weighted_group(rng: np.random.Generator, nodes: np.ndarray,
                    size: int, node_use: np.ndarray, case: str) -> Edge:
    if size > len(nodes):
        raise ValueError("block does not have enough nodes for requested edge size")
    if case == "S1":
        weights = 1.0 / np.square(1.0 + node_use[nodes])
    else:
        weights = 1.0 + 5.0 * node_use[nodes]
    weights /= weights.sum()
    chosen = rng.choice(nodes, size=size, replace=False, p=weights)
    return tuple(sorted(int(i) for i in chosen))


def _add_observation_noise(A: sparse.csr_matrix, rng: np.random.Generator,
                           add_p: float, drop_p: float) -> sparse.csr_matrix:
    if add_p == 0 and drop_p == 0:
        return A.copy()
    n = A.shape[0]
    upper = sparse.triu(A, k=1).tocoo()
    original_pairs = {(int(i), int(j)) for i, j in zip(upper.row, upper.col)}
    kept = rng.random(upper.nnz) >= drop_p
    pairs = {(int(i), int(j)) for i, j, keep in zip(upper.row, upper.col, kept) if keep}
    if add_p:
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in original_pairs and rng.random() < add_p:
                    pairs.add((i, j))
    if not pairs:
        return sparse.csr_matrix((n, n), dtype=np.int8)
    row = [i for i, j in pairs] + [j for i, j in pairs]
    col = [j for i, j in pairs] + [i for i, j in pairs]
    return sparse.csr_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(n, n))


def generate_synthetic(config: SyntheticConfig, seed: int) -> SyntheticSample:
    """Generate one independent S1 or S2 hypergraph from a fixed, stated model.

    S1 uses low-overlap planted groups plus pair-event triangles that are false
    size-3 hyperedges. S2 favors repeated group membership within blocks,
    producing dense overlapping cliques and false subcliques. The real event
    list is never supplied to candidate generation.
    """
    if not isinstance(config, SyntheticConfig):
        raise TypeError("config must be SyntheticConfig")
    rng = np.random.default_rng(seed)
    n = config.n_nodes
    block = rng.permutation(np.arange(n) % config.n_blocks)
    block_nodes = [np.flatnonzero(block == b) for b in range(config.n_blocks)]
    possible_blocks = [b for b, nodes in enumerate(block_nodes)
                       if len(nodes) >= min(config.edge_sizes)]
    if not possible_blocks:
        raise ValueError("no block has enough nodes for the smallest edge size")
    node_use = np.zeros(n, dtype=int)
    events: list[Edge] = []
    event_set: set[Edge] = set()

    # S1 adds explicit false-triangle candidates without adding a true triad.
    n_decoy_triangles = config.num_hyperedges // 8 if config.case == "S1" else 0
    n_group_events = config.num_hyperedges - 3 * n_decoy_triangles
    attempts = 0
    max_attempts = max(5000, 200 * n_group_events)
    while len(events) < n_group_events and attempts < max_attempts:
        attempts += 1
        k = int(rng.choice(config.edge_sizes))
        eligible = [b for b in possible_blocks if len(block_nodes[b]) >= k]
        if not eligible:
            raise ValueError("no block can fit every selected edge size")
        chosen_block = int(rng.choice(eligible))
        edge = _weighted_group(rng, block_nodes[chosen_block], k, node_use,
                               config.case)
        if edge in event_set:
            continue
        events.append(edge)
        event_set.add(edge)
        node_use[list(edge)] += 1
    if len(events) != n_group_events:
        raise RuntimeError("unique hyperedges exhausted; lower count or edge size")

    if n_decoy_triangles:
        decoys_added = 0
        attempts = 0
        while decoys_added < n_decoy_triangles and attempts < 5000:
            attempts += 1
            if config.n_blocks >= 3:
                chosen_blocks = rng.choice(config.n_blocks, size=3, replace=False)
                nodes = [int(rng.choice(block_nodes[int(b)])) for b in chosen_blocks]
            else:
                nodes = list(map(int, rng.choice(n, size=3, replace=False)))
            triple = tuple(sorted(nodes))
            pairs = [(triple[0], triple[1]), (triple[0], triple[2]),
                     (triple[1], triple[2])]
            if triple in event_set or any(pair in event_set for pair in pairs):
                continue
            events.extend(pairs)
            event_set.update(pairs)
            node_use[list(triple)] += 2
            decoys_added += 1
        if decoys_added != n_decoy_triangles:
            raise RuntimeError("could not place the requested false-triangle decoys")

    A_exact = project_hypergraph(events, n, mode="binary")
    A_observed = _add_observation_noise(A_exact, rng, config.edge_add_prob,
                                        config.edge_drop_prob)
    W = (project_hypergraph(events, n, mode="multiplicity")
         if config.include_multiplicity else None)
    X = None
    if config.feature_signal > 0:
        block_centers = np.eye(config.n_blocks)
        X = config.feature_signal * block_centers[block] + rng.normal(
            size=(n, config.n_blocks))
    params = asdict(config)
    params["seed"] = int(seed)
    params["generated_event_count"] = len(events)
    params["decoy_triangle_count"] = n_decoy_triangles
    params["true_unique_edge_count"] = len(event_set)
    record = GraphRecord(
        graph_id=f"synthetic-{config.case.lower()}-{seed}",
        num_nodes=n,
        adjacency_csr=A_observed,
        features=X,
        hyperedges_truth=events,
        hyperedge_event_ids=[f"event-{i:06d}" for i in range(len(events))],
        observation_mode="binary",
        provenance={"dataset": f"synthetic-{config.case}",
                    "generator": "qhyper.synthetic.generate_synthetic",
                    "parameters": params,
                    "observation_noisy": bool(config.edge_add_prob or config.edge_drop_prob)},
    )
    return SyntheticSample(record, W, params, int(seed))
