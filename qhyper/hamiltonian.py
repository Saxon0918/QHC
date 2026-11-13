"""Real symmetric Hamiltonians for quantum-inspired graph dynamics.

The primary model is the normalized graph Laplacian divided by two.  An
isolated vertex has a zero row and diagonal by default, as specified in the
research protocol.  Feature-derived and adjacency alternatives are explicit
ablations; the primary reconstruction experiment uses ``lam=0``.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree


def checked_adjacency(adjacency: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    """Return a validated, loop-free, nonnegative symmetric CSR adjacency.

    Minor floating-point asymmetry is symmetrized, but substantive asymmetry,
    nonfinite values, negative weights and self-loops are rejected rather than
    silently changing the observation model.
    """
    raw = sparse.csr_matrix(adjacency)
    if np.iscomplexobj(raw.data) and np.any(np.imag(raw.data) != 0):
        raise ValueError("adjacency must be real")
    a = (raw.real if np.iscomplexobj(raw.data) else raw).astype(np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("adjacency must be a nonempty square matrix")
    a.sum_duplicates()
    a.eliminate_zeros()
    if not np.isfinite(a.data).all():
        raise ValueError("adjacency contains nonfinite weights")
    if np.any(a.data < 0):
        raise ValueError("adjacency contains negative weights")
    if np.any(np.abs(a.diagonal()) > 1e-12):
        raise ValueError("adjacency must have no self-loops")
    asymmetry = a - a.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1e-10:
        raise ValueError("adjacency must be symmetric")
    a = ((a + a.T) * 0.5).tocsr()
    a.eliminate_zeros()
    return a


def normalized_laplacian(
    adjacency: sparse.spmatrix | np.ndarray,
    *,
    isolated_node_diagonal: float = 0.0,
) -> sparse.csr_matrix:
    """Compute ``D_nonisolated - D^-1/2 A D^-1/2`` as a sparse CSR matrix.

    ``D_nonisolated`` has unit diagonal on positive-degree vertices.  Its
    isolated-vertex diagonal is zero by default, matching Section 4.1 of the
    research specification.  The optional value 1 reproduces the alternative
    common numerical convention and must be recorded if used in an ablation.
    """
    if isolated_node_diagonal not in (0.0, 1.0):
        raise ValueError("isolated_node_diagonal must be 0 or 1")
    a = checked_adjacency(adjacency)
    degree = np.asarray(a.sum(axis=1)).ravel()
    nonisolated = degree > 0
    inverse_root = np.zeros_like(degree)
    inverse_root[nonisolated] = 1.0 / np.sqrt(degree[nonisolated])
    scaling = sparse.diags(inverse_root, format="csr")
    diagonal = np.where(nonisolated, 1.0, isolated_node_diagonal)
    laplacian = sparse.diags(diagonal, format="csr") - scaling @ a @ scaling
    laplacian = laplacian.tocsr()
    laplacian.eliminate_zeros()
    return laplacian


def feature_similarity_adjacency(
    features: np.ndarray,
    *,
    k: int = 10,
) -> sparse.csr_matrix:
    """Build a sparse mutual-kNN cosine graph from allowed node features.

    Only strictly positive cosine similarities are retained.  Zero-norm rows
    remain isolated.  This is an optional feature ablation, never an implicit
    part of the topology-only primary Hamiltonian.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must be an N-by-d matrix")
    if not np.isfinite(x).all():
        raise ValueError("features contain nonfinite values")
    if k < 1:
        raise ValueError("k must be positive")
    n = x.shape[0]
    norms = np.linalg.norm(x, axis=1)
    valid = np.flatnonzero(norms > 0)
    if valid.size < 2:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    unit = x[valid] / norms[valid, None]
    neighbors = min(int(k) + 1, valid.size)
    # Euclidean nearest neighbors on unit vectors rank by cosine similarity.
    _, local_indices = cKDTree(unit).query(unit, k=neighbors)
    local_indices = np.atleast_2d(local_indices)
    if local_indices.shape[0] != valid.size:
        local_indices = local_indices.T
    directed: dict[tuple[int, int], float] = {}
    for local_i, row in enumerate(local_indices):
        i = int(valid[local_i])
        for local_j in np.atleast_1d(row):
            j = int(valid[int(local_j)])
            if i == j:
                continue
            weight = float(unit[local_i] @ unit[int(local_j)])
            if weight > 0:
                directed[i, j] = weight
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for (i, j), weight in directed.items():
        if i < j and (j, i) in directed:
            mutual_weight = min(weight, directed[j, i])
            rows.extend((i, j))
            cols.extend((j, i))
            values.extend((mutual_weight, mutual_weight))
    return sparse.csr_matrix((values, (rows, cols)), shape=(n, n))


def _spectral_radius(matrix: sparse.csr_matrix) -> float:
    if matrix.nnz == 0:
        return 0.0
    n = matrix.shape[0]
    if n <= 32:
        return float(np.max(np.abs(np.linalg.eigvalsh(matrix.toarray()))))
    return float(abs(eigsh(matrix, k=1, which="LM", return_eigenvectors=False)[0]))


def build_hamiltonian(
    adjacency: sparse.spmatrix | np.ndarray,
    features: np.ndarray | None = None,
    *,
    kind: str = "norm_laplacian",
    lam: float = 0.0,
    isolated_node_diagonal: float = 0.0,
    feature_k: int = 10,
) -> sparse.csr_matrix:
    """Construct an explicitly selected real symmetric Hamiltonian.

    ``norm_laplacian`` is ``[(1-lam)L_A + lam L_X]/2``.  ``adjacency_scaled``
    and ``normalized_adjacency`` are separate topology-only ablations.  The
    former divides by the adjacency spectral radius, using zero for an empty
    graph.  No labels are accepted by this interface.
    """
    if not 0.0 <= lam <= 1.0 or not np.isfinite(lam):
        raise ValueError("lam must be finite and in [0, 1]")
    a = checked_adjacency(adjacency)
    if kind == "norm_laplacian":
        la = normalized_laplacian(a, isolated_node_diagonal=isolated_node_diagonal)
        if lam == 0.0:
            return (0.5 * la).tocsr()
        if features is None:
            raise ValueError("features are required when lam > 0")
        x = np.asarray(features)
        if x.ndim != 2 or x.shape[0] != a.shape[0]:
            raise ValueError("feature row count must equal graph node count")
        ax = feature_similarity_adjacency(x, k=feature_k)
        lx = normalized_laplacian(ax, isolated_node_diagonal=isolated_node_diagonal)
        return (0.5 * ((1.0 - lam) * la + lam * lx)).tocsr()
    if lam != 0.0:
        raise ValueError("lam > 0 applies only to norm_laplacian")
    if kind == "adjacency_scaled":
        radius = _spectral_radius(a)
        return (a / radius).tocsr() if radius else a
    if kind == "normalized_adjacency":
        degree = np.asarray(a.sum(axis=1)).ravel()
        inverse_root = np.zeros_like(degree)
        positive = degree > 0
        inverse_root[positive] = 1.0 / np.sqrt(degree[positive])
        d = sparse.diags(inverse_root, format="csr")
        return (d @ a @ d).tocsr()
    raise ValueError(f"unknown Hamiltonian kind: {kind}")
