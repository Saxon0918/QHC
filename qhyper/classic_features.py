"""Classical controls for candidate hyperedge discrimination.

All features in this module use only the observed graph and optional node
attributes. They never inspect hyperedge truth or node labels. The precise
short-time coefficient is included in C0 so that an apparent quantum gain
cannot be explained by an omitted common-neighbor statistic.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


_FEATURE_NAMES = (
    "size",
    "degree_mean",
    "degree_max",
    "degree_variance",
    "internal_edges",
    "internal_density",
    "boundary_weight",
    "candidate_volume",
    "conductance",
    "external_neighbor_count",
    "external_common_neighbor_sum",
    "external_common_neighbor_mean",
    "external_common_neighbor_max",
    "external_coupling_inner_product_sum",
    "short_time_c2_coefficient",
    "internal_triangle_count",
    "neighbor_jaccard_mean",
    "feature_cosine_mean",
    "feature_cosine_min",
    "feature_variance_mean",
    "feature_centroid_norm",
    "source_local_clique",
    "source_maximal_clique",
    "source_maximal_clique_subset",
    "source_structural_sampling",
    "source_feature",
    "source_edge",
    "source_other",
)

_SOURCE_NAMES = (
    "local_clique",
    "maximal_clique",
    "maximal_clique_subset",
    "structural_sampling",
    "feature",
    "edge",
    "other",
)


def classical_feature_names() -> tuple[str, ...]:
    """Return the fixed feature order used for every candidate."""
    return _FEATURE_NAMES


def _adjacency(A: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    graph = sparse.csr_matrix(A, dtype=np.float64)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("A must be square")
    if graph.data.size and (not np.all(np.isfinite(graph.data)) or np.min(graph.data) < 0):
        raise ValueError("A must have finite nonnegative weights")
    if np.any(np.abs(graph.diagonal()) > 1e-12):
        raise ValueError("A must have a zero diagonal")
    delta = graph - graph.T
    if delta.data.size and np.max(np.abs(delta.data)) > 1e-10:
        raise ValueError("A must be symmetric")
    graph.eliminate_zeros()
    return graph


def _nodes(candidate: Iterable[int], n: int) -> np.ndarray:
    nodes = np.asarray(tuple(candidate), dtype=np.int64)
    if nodes.ndim != 1 or nodes.size < 2:
        raise ValueError("candidate must contain at least two node IDs")
    if np.any(nodes < 0) or np.any(nodes >= n) or np.unique(nodes).size != nodes.size:
        raise ValueError("candidate node IDs must be unique and in range")
    return np.sort(nodes)


def _normalized_laplacian(graph: sparse.csr_matrix) -> sparse.csr_matrix:
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt[nonzero] = degree[nonzero] ** -0.5
    normalized = sparse.diags(inv_sqrt) @ graph @ sparse.diags(inv_sqrt)
    # Isolated nodes use L_ii=0, matching the primary Hamiltonian convention.
    return (sparse.diags(nonzero.astype(float)) - normalized).tocsr()


def short_time_coefficient(
    A: sparse.spmatrix | np.ndarray,
    candidate: Iterable[int],
) -> float:
    """Exact coefficient of t^2 in C(e,t) for H=L_A/2.

    It is -2/k times the sum of pairwise inner products of the external
    coupling columns H[e^c,e]. This is formula (3) in the research plan.
    """
    graph = _adjacency(A)
    nodes = _nodes(candidate, graph.shape[0])
    n = graph.shape[0]
    outside = np.setdiff1d(np.arange(n), nodes, assume_unique=True)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree)
    good = degree > 0
    inv_sqrt[good] = degree[good] ** -0.5
    external = graph[outside, :][:, nodes]
    columns = -0.5 * (sparse.diags(inv_sqrt[outside]) @ external @ sparse.diags(inv_sqrt[nodes]))
    gram = np.asarray((columns.T @ columns).toarray())
    return float(-(np.sum(gram) - np.trace(gram)) / len(nodes))


def classical_candidate_features(
    A: sparse.spmatrix | np.ndarray,
    candidate: Iterable[int],
    X: np.ndarray | None = None,
    *,
    source: str | Iterable[str] | None = None,
) -> np.ndarray:
    """Extract C0 features in a stable order from one observed candidate.

    The same dimensions are returned with or without attributes; absent
    attribute statistics are zero. Proposal source is encoded with a fixed
    vocabulary shared by all splits, avoiding test-fitted categories.
    """
    graph = _adjacency(A)
    return _candidate_features_prepared(graph, candidate, X, source)


def _candidate_features_prepared(
    graph: sparse.csr_matrix,
    candidate: Iterable[int],
    X: np.ndarray | None,
    source: str | Iterable[str] | None,
) -> np.ndarray:
    n = graph.shape[0]
    nodes = _nodes(candidate, n)
    k = nodes.size
    degree = np.asarray(graph.sum(axis=1)).ravel()
    internal = graph[nodes, :][:, nodes]
    internal_weight = float(internal.sum()) / 2.0
    internal_density = float(internal.nnz) / (k * (k - 1))
    volume = float(degree[nodes].sum())
    boundary = volume - 2 * internal_weight
    remaining_volume = float(degree.sum()) - volume
    denominator = min(volume, remaining_volume)
    conductance = boundary / denominator if denominator > 0 else 0.0

    outside = np.setdiff1d(np.arange(n), nodes, assume_unique=True)
    external = graph[outside, :][:, nodes].tocsr()
    external_bin = external.copy()
    external_bin.data[:] = 1.0
    external_gram = np.asarray((external_bin.T @ external_bin).toarray())
    pair_values = external_gram[np.triu_indices(k, 1)]
    external_common_sum = float(pair_values.sum())
    external_common_mean = float(pair_values.mean()) if pair_values.size else 0.0
    external_common_max = float(pair_values.max()) if pair_values.size else 0.0
    external_neighbor_count = int(np.count_nonzero(np.asarray(external_bin.sum(axis=1)).ravel()))

    inv_sqrt = np.zeros_like(degree)
    good = degree > 0
    inv_sqrt[good] = degree[good] ** -0.5
    coupling = -0.5 * (sparse.diags(inv_sqrt[outside]) @ external @ sparse.diags(inv_sqrt[nodes]))
    gram = np.asarray((coupling.T @ coupling).toarray())
    coupling_sum = float(np.sum(np.triu(gram, 1)))
    c2 = float(-2.0 * coupling_sum / k)

    internal_binary = (internal.toarray() > 0).astype(float)
    triangles = float(np.trace(internal_binary @ internal_binary @ internal_binary) / 6.0)
    neighbor_sets = [set(graph.getrow(int(node)).indices) for node in nodes]
    jaccards = []
    for a, b in combinations(range(k), 2):
        union = neighbor_sets[a] | neighbor_sets[b]
        jaccards.append(len(neighbor_sets[a] & neighbor_sets[b]) / len(union) if union else 0.0)

    cos_mean = cos_min = feature_variance = centroid_norm = 0.0
    if X is not None:
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] != n or not np.all(np.isfinite(features)):
            raise ValueError("X must be a finite (N,d) array")
        selected = features[nodes]
        norms = np.linalg.norm(selected, axis=1)
        similarities = []
        for a, b in combinations(range(k), 2):
            similarities.append(float(np.dot(selected[a], selected[b]) / (norms[a] * norms[b])) if norms[a] > 0 and norms[b] > 0 else 0.0)
        cos_mean = float(np.mean(similarities)) if similarities else 0.0
        cos_min = float(np.min(similarities)) if similarities else 0.0
        feature_variance = float(np.mean(np.var(selected, axis=0)))
        centroid_norm = float(np.linalg.norm(np.mean(selected, axis=0)))

    if source is None:
        source_names: set[str] = set()
    elif isinstance(source, str):
        source_names = {source}
    else:
        source_names = set(source)
        if not all(isinstance(name, str) for name in source_names):
            raise ValueError("proposal sources must be strings")
    aliases = {
        "local_expansion": "local_clique",
        "maximal_clique_subsets": "maximal_clique_subset",
        "feature_knn": "feature",
    }
    normalized_sources = {aliases.get(name, name) for name in source_names}
    if any(name not in _SOURCE_NAMES for name in normalized_sources):
        normalized_sources.add("other")
    source_values = [float(category in normalized_sources) for category in _SOURCE_NAMES]

    result = np.array(
        [
            k,
            np.mean(degree[nodes]),
            np.max(degree[nodes]),
            np.var(degree[nodes]),
            internal_weight,
            internal_density,
            boundary,
            volume,
            conductance,
            external_neighbor_count,
            external_common_sum,
            external_common_mean,
            external_common_max,
            coupling_sum,
            c2,
            triangles,
            np.mean(jaccards) if jaccards else 0.0,
            cos_mean,
            cos_min,
            feature_variance,
            centroid_norm,
            *source_values,
        ],
        dtype=np.float64,
    )
    assert result.shape == (len(_FEATURE_NAMES),)
    return result


def classical_candidate_matrix(
    A: sparse.spmatrix | np.ndarray,
    candidates: Iterable[Iterable[int]],
    X: np.ndarray | None = None,
    *,
    sources: Iterable[str | Iterable[str] | None] | None = None,
) -> np.ndarray:
    """Batch C0 extraction with one adjacency validation for a shared graph."""
    graph = _adjacency(A)
    candidate_list = list(candidates)
    source_list = [None] * len(candidate_list) if sources is None else list(sources)
    if len(source_list) != len(candidate_list):
        raise ValueError("sources must align with candidates")
    if not candidate_list:
        return np.empty((0, len(_FEATURE_NAMES)), dtype=float)
    return np.vstack(
        [_candidate_features_prepared(graph, edge, X, source) for edge, source in zip(candidate_list, source_list)]
    )


def family_feature_names(family: str, times: Sequence[float]) -> tuple[str, ...]:
    """Return names for C1/C2/C3/Q1 control vectors."""
    if family == "spectral":
        return tuple(f"walk_r{r}_{name}" for r in (2, 3, 4) for name in ("offdiag_mean", "internal_frobenius_mean"))
    if family in {"heat", "wave"}:
        return tuple(f"{family}_t{t:g}_{name}" for t in times for name in ("coherent_localization", "incoherent_localization", "contrast"))
    if family == "quantum_pair":
        return tuple(f"quantum_pair_t{t:g}_{name}" for t in times for name in ("offdiag_mean", "offdiag_max", "offdiag_std", "diagonal_mean", "internal_mean"))
    raise ValueError(f"unknown family: {family}")


def _normalized_localization(acted: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    norms_sq = np.sum(np.abs(acted) ** 2, axis=0)
    if np.any(norms_sq <= 1e-24):
        raise FloatingPointError("classical response has near-zero norm; choose another time")
    return np.sum(np.abs(acted[nodes, :]) ** 2, axis=0) / norms_sq


def dynamic_family_features(
    A: sparse.spmatrix | np.ndarray,
    H: sparse.spmatrix | np.ndarray | None,
    candidate: Iterable[int],
    times: Sequence[float],
    family: str,
) -> np.ndarray:
    """Compute matched C1 heat, C2 wave, C3 walks, or Q1 pairwise controls.

    Heat and wave responses are globally norm-normalized localization of
    identically normalized coherent/basis inputs. They are *classical*
    dynamics statistics, not unitary Born probabilities. The C3 normalized
    adjacency-walk control uses orders 2, 3 and 4 and ignores ``times``.
    Q1 aggregates |U_vi|^2 within the candidate, without group coherence.
    """
    graph = _adjacency(A)
    n = graph.shape[0]
    nodes = _nodes(candidate, n)
    k = nodes.size
    if family == "spectral":
        degree = np.asarray(graph.sum(axis=1)).ravel()
        inv_sqrt = np.zeros_like(degree)
        good = degree > 0
        inv_sqrt[good] = degree[good] ** -0.5
        walk = (sparse.diags(inv_sqrt) @ graph @ sparse.diags(inv_sqrt)).tocsr()
        action = sparse.csr_matrix((np.ones(k), (nodes, np.arange(k))), shape=(n, k))
        values = []
        for order in range(1, 5):
            action = walk @ action
            if order >= 2:
                block = np.asarray(action[nodes, :].toarray())
                offdiag = block[~np.eye(k, dtype=bool)]
                values.extend([float(np.mean(offdiag)), float(np.mean(block ** 2))])
        return np.asarray(values, dtype=float)

    if len(times) == 0 or any(not np.isfinite(t) or t < 0 for t in times):
        raise ValueError("times must be a nonempty sequence of finite nonnegative values")
    basis = np.zeros((n, k + 1), dtype=float)
    basis[nodes, 0] = 1.0 / np.sqrt(k)
    basis[nodes, np.arange(1, k + 1)] = 1.0
    L = _normalized_laplacian(graph)
    if family == "wave":
        zero = sparse.csr_matrix((n, n))
        wave_generator = sparse.bmat([[zero, sparse.eye(n)], [-L, zero]], format="csr")
        wave_initial = np.vstack((basis, np.zeros_like(basis)))
    elif family == "quantum_pair":
        if H is None:
            raise ValueError("H is required for quantum_pair")
        H = sparse.csr_matrix(H, dtype=float)
        if H.shape != (n, n):
            raise ValueError("H shape must match A")
        delta = H - H.T
        if delta.data.size and np.max(np.abs(delta.data)) > 1e-10:
            raise ValueError("H must be symmetric")
    elif family != "heat":
        raise ValueError(f"unknown family: {family}")

    values = []
    for t in times:
        if family == "heat":
            acted = np.asarray(expm_multiply((-float(t)) * L, basis))
        elif family == "wave":
            acted = np.asarray(expm_multiply(float(t) * wave_generator, wave_initial))[:n, :]
        else:
            acted = np.asarray(expm_multiply((-1j * float(t)) * H, basis[:, 1:]))
        if family in {"heat", "wave"}:
            localization = _normalized_localization(acted, nodes)
            coherent = float(localization[0])
            incoherent = float(np.mean(localization[1:]))
            values.extend([coherent, incoherent, coherent - incoherent])
        else:
            block = np.abs(acted[nodes, :]) ** 2
            offdiag = block[~np.eye(k, dtype=bool)]
            values.extend(
                [
                    float(np.mean(offdiag)),
                    float(np.max(offdiag)),
                    float(np.std(offdiag)),
                    float(np.mean(np.diag(block))),
                    float(np.mean(block)),
                ]
            )
    return np.asarray(values, dtype=float)


def dynamic_family_matrix(
    A: sparse.spmatrix | np.ndarray,
    H: sparse.spmatrix | np.ndarray | None,
    candidates: Iterable[Iterable[int]],
    times: Sequence[float],
    family: str,
    *,
    max_state_columns: int = 64,
) -> np.ndarray:
    """Batch C1/C2/C3/Q1 extraction without a dense full-graph propagator.

    Heat/wave propagate bounded batches of candidate coherent and member
    basis states. Spectral walks reuse the normalized adjacency. Q1 evolves
    each distinct candidate member basis state once per time point, then
    gathers the required candidate-local probability blocks. Every temporary
    propagated matrix has at most ``max_state_columns`` columns.
    """
    graph = _adjacency(A)
    n = graph.shape[0]
    edges = [_nodes(edge, n) for edge in candidates]
    m = len(edges)
    if not isinstance(max_state_columns, int) or max_state_columns < 1:
        raise ValueError("max_state_columns must be a positive integer")
    if family not in {"heat", "wave", "spectral", "quantum_pair"}:
        raise ValueError(f"unknown family: {family}")
    if family != "spectral" and (
        len(times) == 0 or any(not np.isfinite(t) or t < 0 for t in times)
    ):
        raise ValueError("times must be a nonempty sequence of finite nonnegative values")
    if not m:
        return np.empty((0, len(family_feature_names(family, times))), dtype=float)

    if family == "quantum_pair":
        if H is None:
            raise ValueError("H is required for quantum_pair")
        h = sparse.csr_matrix(H, dtype=float)
        if h.shape != (n, n):
            raise ValueError("H shape must match A")
        delta = h - h.T
        if delta.data.size and np.max(np.abs(delta.data)) > 1e-10:
            raise ValueError("H must be symmetric")
        unique_nodes = sorted({int(node) for edge in edges for node in edge})
        occurrences: dict[int, list[tuple[int, int]]] = {node: [] for node in unique_nodes}
        for edge_index, edge in enumerate(edges):
            for member_position, node in enumerate(edge):
                occurrences[int(node)].append((edge_index, member_position))
        result = np.empty((m, 5 * len(times)), dtype=float)
        for time_index, time in enumerate(times):
            blocks = [np.empty((len(edge), len(edge)), dtype=float) for edge in edges]
            for start in range(0, len(unique_nodes), max_state_columns):
                node_batch = unique_nodes[start : start + max_state_columns]
                basis = np.zeros((n, len(node_batch)), dtype=float)
                basis[node_batch, np.arange(len(node_batch))] = 1.0
                evolved = np.asarray(expm_multiply((-1j * float(time)) * h, basis))
                for column, node in enumerate(node_batch):
                    for edge_index, member_position in occurrences[node]:
                        blocks[edge_index][:, member_position] = (
                            np.abs(evolved[edges[edge_index], column]) ** 2
                        )
            for edge_index, block in enumerate(blocks):
                offdiag = block[~np.eye(len(block), dtype=bool)]
                result[edge_index, 5 * time_index : 5 * time_index + 5] = [
                    float(np.mean(offdiag)),
                    float(np.max(offdiag)),
                    float(np.std(offdiag)),
                    float(np.mean(np.diag(block))),
                    float(np.mean(block)),
                ]
        return result

    if family == "spectral":
        degree = np.asarray(graph.sum(axis=1)).ravel()
        inv_sqrt = np.zeros_like(degree)
        nonzero = degree > 0
        inv_sqrt[nonzero] = degree[nonzero] ** -0.5
        walk = (sparse.diags(inv_sqrt) @ graph @ sparse.diags(inv_sqrt)).tocsr()
        result = np.empty((m, 6), dtype=float)
        for batch in _candidate_batches(edges, max_state_columns, coherent=False):
            states, offsets = _make_state_block(n, edges, batch, coherent=False)
            action = sparse.csr_matrix(states)
            for order in range(1, 5):
                action = walk @ action
                if order < 2:
                    continue
                for edge_index, start, stop in offsets:
                    edge = edges[edge_index]
                    block = np.asarray(action[edge, start:stop].toarray())
                    offdiag = block[~np.eye(len(edge), dtype=bool)]
                    result[edge_index, 2 * (order - 2) : 2 * (order - 2) + 2] = [
                        float(np.mean(offdiag)), float(np.mean(block ** 2))
                    ]
        return result

    laplacian = _normalized_laplacian(graph)
    if family == "wave":
        zero = sparse.csr_matrix((n, n))
        generator = sparse.bmat([[zero, sparse.eye(n)], [-laplacian, zero]], format="csr")
    result = np.empty((m, 3 * len(times)), dtype=float)
    for batch in _candidate_batches(edges, max_state_columns, coherent=True):
        states, offsets = _make_state_block(n, edges, batch, coherent=True)
        if family == "wave":
            initial = np.vstack((states, np.zeros_like(states)))
        for time_index, time in enumerate(times):
            if family == "heat":
                evolved = np.asarray(expm_multiply((-float(time)) * laplacian, states))
            else:
                evolved = np.asarray(expm_multiply(float(time) * generator, initial))[:n, :]
            for edge_index, start, stop in offsets:
                edge = edges[edge_index]
                localized = _normalized_localization(evolved[:, start:stop], edge)
                coherent_value = float(localized[0])
                incoherent_value = float(np.mean(localized[1:]))
                result[edge_index, 3 * time_index : 3 * time_index + 3] = [
                    coherent_value,
                    incoherent_value,
                    coherent_value - incoherent_value,
                ]
    return result


def _candidate_batches(
    edges: list[np.ndarray], max_state_columns: int, *, coherent: bool
) -> Iterable[list[int]]:
    batch: list[int] = []
    columns = 0
    for index, edge in enumerate(edges):
        needed = len(edge) + int(coherent)
        if needed > max_state_columns:
            raise ValueError("max_state_columns must fit one candidate's states")
        if batch and columns + needed > max_state_columns:
            yield batch
            batch = []
            columns = 0
        batch.append(index)
        columns += needed
    if batch:
        yield batch


def _make_state_block(
    n: int, edges: list[np.ndarray], batch: list[int], *, coherent: bool
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    width = sum(len(edges[index]) + int(coherent) for index in batch)
    block = np.zeros((n, width), dtype=float)
    offsets: list[tuple[int, int, int]] = []
    cursor = 0
    for index in batch:
        edge = edges[index]
        start = cursor
        if coherent:
            block[edge, cursor] = 1.0 / np.sqrt(len(edge))
            cursor += 1
        block[edge, cursor + np.arange(len(edge))] = 1.0
        cursor += len(edge)
        offsets.append((index, start, cursor))
    return block, offsets
