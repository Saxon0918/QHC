"""Covariance-certified clustering of quantum/Markov measurement outcomes.

For the real-symmetric transport matrices used in this project, ``P`` is
symmetric and doubly stochastic.  Write its columns as probability vectors
``p_r``.  Their Gram kernel is ``K=P.T @ P=Σ_r p_r p_r.T``.  A partition of
measurement outcomes into clusters ``g`` defines one soft hyperedge per
cluster, with incidence ``b_g=Σ_{r in g}p_r`` and edge degree ``|g|``.
The standard HGNN operator becomes

    Θ = Σ_g b_g b_g.T / |g|.

For ``μ_g=b_g/|g|``, the exact difference is

    K-Θ = Σ_g Σ_{r in g}(p_r-μ_g)(p_r-μ_g).T >= 0.

Consequently its spectral error is at most its trace, which is exactly the
within-cluster squared error.  This certifies operator approximation only;
it says nothing about recovering event hyperedges or node classification.
Ward agglomeration greedily minimizes each next increase in this trace.  It
does not globally minimize the trace among all m-way partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class ClusteredTransportHypergraph:
    incidence: sparse.csr_matrix
    normalized_theta: sparse.csr_matrix
    clusters: tuple[tuple[int, ...], ...]
    cluster_sizes: np.ndarray
    trace_error: float
    spectral_error: float
    membership_nnz: int


def _checked_transport(transport: np.ndarray, max_nodes: int) -> np.ndarray:
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    raw = np.asarray(transport)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or raw.shape[0] < 1:
        raise ValueError("transport must be a nonempty square matrix")
    n = raw.shape[0]
    if n > max_nodes:
        raise ValueError(f"dense cluster construction is limited to max_nodes={max_nodes}")
    if np.iscomplexobj(raw):
        raise ValueError("transport must be real")
    try:
        p = np.asarray(raw, dtype=np.float64).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("transport must contain real numeric values") from exc
    if not np.isfinite(p).all() or np.min(p) < -1e-12:
        raise ValueError("transport must be finite and nonnegative")
    np.maximum(p, 0.0, out=p)
    if not np.allclose(p, p.T, rtol=0.0, atol=1e-10):
        raise ValueError("transport must be symmetric; column clustering otherwise targets P P.T")
    if (not np.allclose(p.sum(axis=0), 1.0, rtol=0.0, atol=1e-10)
            or not np.allclose(p.sum(axis=1), 1.0, rtol=0.0, atol=1e-10)):
        raise ValueError("transport must be doubly stochastic")
    return p


def _checked_partition(
    partition: Sequence[Sequence[int]], n: int
) -> tuple[tuple[int, ...], ...]:
    try:
        groups = tuple(tuple(group) for group in partition)
    except TypeError as exc:
        raise ValueError("partition must contain nonempty index groups") from exc
    if not groups or any(not group for group in groups):
        raise ValueError("partition must contain nonempty index groups")
    flattened: list[int] = []
    for group in groups:
        for index in group:
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
                raise ValueError("partition indices must be integers")
            flattened.append(int(index))
    if sorted(flattened) != list(range(n)):
        raise ValueError("partition must cover every outcome exactly once")
    return tuple(sorted((tuple(sorted(int(index) for index in group))
                         for group in groups), key=lambda group: group[0]))


def _ward_partition(p: np.ndarray, n_clusters: int) -> tuple[tuple[int, ...], ...]:
    n = p.shape[0]
    active: dict[int, tuple[tuple[int, ...], np.ndarray]] = {
        index: ((index,), p[:, index].copy()) for index in range(n)
    }
    next_id = n
    while len(active) > n_clusters:
        keys = sorted(active, key=lambda key: active[key][0][0])
        best_key: tuple[float, int, int] | None = None
        best_pair: tuple[int, int] | None = None
        for position, left_id in enumerate(keys):
            left_members, left_sum = active[left_id]
            left_size = len(left_members)
            left_mean = left_sum / left_size
            for right_id in keys[position + 1:]:
                right_members, right_sum = active[right_id]
                right_size = len(right_members)
                difference = left_mean - right_sum / right_size
                merge_cost = (left_size * right_size / (left_size + right_size)
                              * float(difference @ difference))
                # Index tie-breaking is reproducible, but breaks pointwise
                # permutation equivariance when distinct pairs tie exactly.
                key = (merge_cost, left_members[0], right_members[0])
                if best_key is None or key < best_key:
                    best_key = key
                    best_pair = (left_id, right_id)
        if best_pair is None:
            raise RuntimeError("Ward agglomeration could not find a merge")
        left_id, right_id = best_pair
        left_members, left_sum = active.pop(left_id)
        right_members, right_sum = active.pop(right_id)
        members = tuple(sorted(left_members + right_members))
        active[next_id] = (members, left_sum + right_sum)
        next_id += 1
    return tuple(sorted((members for members, _ in active.values()),
                        key=lambda members: members[0]))


def cluster_transport_hypergraph(
    transport: np.ndarray,
    *,
    n_clusters: int | None = None,
    partition: Sequence[Sequence[int]] | None = None,
    max_nodes: int = 128,
) -> ClusteredTransportHypergraph:
    """Compress all measurement endpoints into ``n_clusters`` soft groups.

    Provide exactly one of ``n_clusters`` (deterministic Ward agglomeration)
    or a validated ``partition``.  Tie-free Ward decisions commute with node
    relabeling; when merge costs tie, node-index tie-breaking is deterministic
    but not pointwise permutation-equivariant.
    """
    p = _checked_transport(transport, max_nodes)
    n = p.shape[0]
    if (n_clusters is None) == (partition is None):
        raise ValueError("provide exactly one of n_clusters or partition")
    if partition is not None:
        groups = _checked_partition(partition, n)
    else:
        if (isinstance(n_clusters, bool) or not isinstance(n_clusters, int)
                or n_clusters < 1 or n_clusters > n):
            raise ValueError("n_clusters must be an integer in [1, n]")
        groups = _ward_partition(p, n_clusters)

    sizes = np.asarray([len(group) for group in groups], dtype=np.float64)
    columns = [p[:, group].sum(axis=1) for group in groups]
    incidence_dense = np.column_stack(columns)
    incidence = sparse.csr_matrix(incidence_dense)
    theta = (incidence @ sparse.diags(1.0 / sizes) @ incidence.T).tocsr()
    theta.eliminate_zeros()

    # This direct covariance sum avoids cancellation when m is close to n.
    trace_error = 0.0
    for group, group_sum, size in zip(groups, columns, sizes):
        deviations = p[:, group] - group_sum[:, None] / size
        trace_error += float(np.sum(deviations * deviations))
    kernel = p.T @ p
    difference = kernel - theta.toarray()
    difference = 0.5 * (difference + difference.T)
    eigenvalues = np.linalg.eigvalsh(difference)
    if eigenvalues[0] < -1e-9:
        raise RuntimeError("cluster covariance identity failed numerically")
    spectral_error = max(0.0, float(eigenvalues[-1]))
    if abs(float(np.trace(difference)) - trace_error) > 1e-8 * max(1, n):
        raise RuntimeError("cluster trace identity failed numerically")
    return ClusteredTransportHypergraph(
        incidence=incidence,
        normalized_theta=theta,
        clusters=groups,
        cluster_sizes=sizes,
        trace_error=trace_error,
        spectral_error=spectral_error,
        membership_nnz=int(incidence.nnz),
    )
