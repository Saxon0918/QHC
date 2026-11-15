"""Independent node-task generators for the hypergraph construction study.

The three regimes share the same latent class assignment and node features for
the same seed.  Only the edge mechanism changes.  In ``group``, homogeneous
four-node events are projected to a binary graph.  In ``pairwise``, edges are
sampled uniformly conditional on the *same realized within-class and
cross-class edge counts* as the group graph.  In ``heterophily``, the total
edge count is matched exactly, but cross-class pairs are more likely than
within-class pairs.  Thus density alone cannot explain arm differences.

The class labels are drawn *before* graph construction.  No transport score,
candidate hyperedge, or downstream prediction participates in their creation.
The planted groups in metadata are diagnostic ground truth and must not be
passed to a construction method evaluated on this graph.
"""

from __future__ import annotations

from itertools import combinations
from math import ceil, comb

import numpy as np
from scipy import sparse


_REGIMES = {"group", "pairwise", "heterophily"}
_N_CLASSES = 3
_FEATURE_DIM = 8
_GROUP_SIZE = 4
_GROUPS_PER_NODE = 0.6
_BACKGROUND_WITHIN = 0.03
_BACKGROUND_CROSS = 0.02
_HETEROPHILY_WITHIN_DENSITY_RATIO = 0.25
_FEATURE_SIGNAL = 1.0
_FEATURE_NOISE_STD = 1.4


def _class_sizes(n: int) -> list[int]:
    return [n // _N_CLASSES + int(k < n % _N_CLASSES) for k in range(_N_CLASSES)]


def _group_count(class_size: int) -> int:
    requested = max(ceil(class_size / _GROUP_SIZE), round(_GROUPS_PER_NODE * class_size))
    return min(requested, comb(class_size, _GROUP_SIZE))


def _planted_groups(
    class_nodes: np.ndarray, rng: np.random.Generator
) -> list[tuple[int, ...]]:
    """Cover each class node once, then sample additional distinct groups."""
    nodes = rng.permutation(class_nodes)
    groups: set[tuple[int, ...]] = set()
    for start in range(0, len(nodes), _GROUP_SIZE):
        block = nodes[start : start + _GROUP_SIZE]
        if len(block) < _GROUP_SIZE:
            available = np.setdiff1d(class_nodes, block, assume_unique=True)
            extra = rng.choice(available, size=_GROUP_SIZE - len(block), replace=False)
            block = np.concatenate((block, extra))
        groups.add(tuple(sorted(int(v) for v in block)))
    target = _group_count(len(class_nodes))
    while len(groups) < target:
        group = rng.choice(class_nodes, size=_GROUP_SIZE, replace=False)
        groups.add(tuple(sorted(int(v) for v in group)))
    return sorted(groups)


def _sample_pairs(
    adjacency: np.ndarray,
    possibilities: list[tuple[int, int]],
    count: int,
    rng: np.random.Generator,
) -> None:
    """Uniformly choose a fixed number of unordered pairs without replacement."""
    if count < 0 or count > len(possibilities):
        raise ValueError("matched edge count exceeds available pairs")
    if count:
        for index in rng.choice(len(possibilities), size=count, replace=False):
            u, v = possibilities[int(index)]
            adjacency[u, v] = adjacency[v, u] = 1


def make_synthetic_graph(
    regime: str, seed: int, n: int = 60
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, dict[str, object]]:
    """Draw one reproducible binary graph, features, labels and provenance.

    ``n >= 12`` permits at least four nodes per class.  The resulting graph is
    undirected, loop-free, and unweighted.  All three regimes use the same
    seeded class and feature draw.  The features carry a weak class signal,
    so graph propagation can help or hurt under the different edge regimes.
    """
    if regime not in _REGIMES:
        raise ValueError(f"regime must be one of {sorted(_REGIMES)}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 12:
        raise ValueError("n must be an integer of at least 12")
    n = int(n)
    rng = np.random.default_rng(int(seed))

    labels = np.repeat(np.arange(_N_CLASSES, dtype=np.int64), _class_sizes(n))
    labels = labels[rng.permutation(n)]
    features = rng.normal(0.0, _FEATURE_NOISE_STD, size=(n, _FEATURE_DIM))
    features[np.arange(n), labels] += _FEATURE_SIGNAL

    class_nodes = [np.flatnonzero(labels == k) for k in range(_N_CLASSES)]
    within_possibilities = [list(combinations(nodes.tolist(), 2)) for nodes in class_nodes]
    cross_possibilities = [
        (u, v) for u in range(n) for v in range(u + 1, n) if labels[u] != labels[v]
    ]
    within_pairs = sum(len(pairs) for pairs in within_possibilities)
    reference = np.zeros((n, n), dtype=np.int8)
    groups = [group for nodes in class_nodes for group in _planted_groups(nodes, rng)]
    for group in groups:
        for offset, u in enumerate(group):
            for v in group[offset + 1 :]:
                reference[u, v] = reference[v, u] = 1
    for u in range(n):
        for v in range(u + 1, n):
            p = _BACKGROUND_WITHIN if labels[u] == labels[v] else _BACKGROUND_CROSS
            if rng.random() < p:
                reference[u, v] = reference[v, u] = 1

    reference_within_counts = [
        sum(int(reference[u, v]) for u, v in pairs) for pairs in within_possibilities
    ]
    reference_cross_count = sum(int(reference[u, v]) for u, v in cross_possibilities)
    reference_edges = sum(reference_within_counts) + reference_cross_count
    overall_reference_density = reference_edges / comb(n, 2)
    heterophily_within_target = round(
        _HETEROPHILY_WITHIN_DENSITY_RATIO * overall_reference_density * within_pairs
    )
    heterophily_cross_target = reference_edges - heterophily_within_target
    if heterophily_cross_target > len(cross_possibilities):
        raise ValueError("matched edge count exceeds cross-class pair capacity")
    heterophily_within = heterophily_within_target / within_pairs
    heterophily_cross = heterophily_cross_target / len(cross_possibilities)
    if heterophily_cross <= heterophily_within:
        raise ValueError("heterophily allocation failed for this graph size")

    if regime == "group":
        adjacency = reference
    else:
        adjacency = np.zeros_like(reference)
        if regime == "pairwise":
            for pairs, count in zip(within_possibilities, reference_within_counts):
                _sample_pairs(adjacency, pairs, count, rng)
            _sample_pairs(adjacency, cross_possibilities, reference_cross_count, rng)
        else:
            all_within = [pair for pairs in within_possibilities for pair in pairs]
            _sample_pairs(adjacency, all_within, heterophily_within_target, rng)
            _sample_pairs(adjacency, cross_possibilities, heterophily_cross_target, rng)

    graph = sparse.csr_matrix(adjacency)
    metadata: dict[str, object] = {
        "regime": regime,
        "seed": int(seed),
        "n_nodes": n,
        "n_classes": _N_CLASSES,
        "class_sizes": _class_sizes(n),
        "feature_dim": _FEATURE_DIM,
        "realized_edges": graph.nnz // 2,
        "matched_reference_edges": int(reference_edges),
        "matched_reference_within_by_class": reference_within_counts,
        "matched_reference_cross": int(reference_cross_count),
        "parameters": {
            "group_size": _GROUP_SIZE,
            "groups_per_node": _GROUPS_PER_NODE,
            "background_within": _BACKGROUND_WITHIN,
            "background_cross": _BACKGROUND_CROSS,
            "pairwise_null": "uniform fixed counts by class and cross-class",
            "pairwise_within_marginal_by_class": [
                count / len(pairs)
                for count, pairs in zip(reference_within_counts, within_possibilities)
            ],
            "pairwise_cross_marginal": reference_cross_count / len(cross_possibilities),
            "heterophily_within_density_ratio": _HETEROPHILY_WITHIN_DENSITY_RATIO,
            "heterophily_within": float(heterophily_within),
            "heterophily_cross": float(heterophily_cross),
            "feature_signal": _FEATURE_SIGNAL,
            "feature_noise_std": _FEATURE_NOISE_STD,
        },
        "latent_groups_diagnostic_only": (
            [list(group) for group in sorted(groups)] if regime == "group" else []
        ),
    }
    return graph, features, labels, metadata


def stratified_node_split(
    labels: np.ndarray,
    seed: int,
    train_fraction: float = 0.5,
    validation_fraction: float = 0.25,
) -> dict[str, np.ndarray]:
    """Return disjoint class-stratified train/validation/test node indices.

    Rounding is performed separately within each class; each class must have
    at least three nodes so every role receives at least one member.
    """
    y = np.asarray(labels)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("labels must be a nonempty one-dimensional array")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if (
        not np.isfinite(train_fraction)
        or not np.isfinite(validation_fraction)
        or not 0.0 < train_fraction < 1.0
        or not 0.0 < validation_fraction < 1.0
        or train_fraction + validation_fraction >= 1.0
    ):
        raise ValueError("fractions must be positive with train + validation < 1")

    rng = np.random.default_rng(int(seed))
    roles: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for class_label in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == class_label))
        count = len(members)
        if count < 3:
            raise ValueError("every class needs at least three nodes")
        n_train = max(1, min(count - 2, round(train_fraction * count)))
        n_validation = max(
            1, min(count - n_train - 1, round(validation_fraction * count))
        )
        roles["train"].extend(members[:n_train].tolist())
        roles["validation"].extend(members[n_train : n_train + n_validation].tolist())
        roles["test"].extend(members[n_train + n_validation :].tolist())
    return {name: np.asarray(sorted(indices), dtype=np.int64) for name, indices in roles.items()}
