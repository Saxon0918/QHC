"""Mass-conserving sparse soft hypergraphs from selected transport columns.

Let ``p_r`` be a probability vector for measuring quantum-walk endpoint ``r``
from every initial vertex (or the analogous symmetric Markov distribution).
The complete transport kernel is ``K = Σ_r p_r p_r.T``.  For selected endpoint
columns, retain only a small support ``b_r`` of each ``p_r`` and assign all
unretained probability mass to singleton hyperedges.  The resulting standard
HGNN operator is

    Θ = Σ_r b_r b_r.T / sum(b_r) + diag(1 - Σ_r b_r).

If the supplied columns are a subset of a symmetric doubly stochastic
transport matrix, Θ is nonnegative, doubly stochastic and positive
semidefinite.  Jensen's inequality gives ``Θ - K >= 0`` in Loewner order.
Thus ``||Θ-K||_2 <= tr(Θ-K)``.  The returned ``trace_bound`` equals this trace
for all endpoints, and is a valid upper bound when endpoints are omitted.
For one endpoint and a fixed number ``k`` of retained nodes, the decrease in
trace error is ``g(S)=s-||b||²/s`` with ``s=sum(b)``.  Its derivative with
respect to any retained probability is
``1-2(b_i/s)+||b||²/s² >= (1-b_i/s)² >= 0``; therefore keeping the ``k``
largest probabilities maximizes this decrease.  This is a local trace
optimality statement, not a global downstream-performance guarantee.
It certifies *linear operator approximation*, not downstream accuracy or
recovery of recorded hyperedges.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as integer_index
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


@dataclass(frozen=True)
class SparseTransportHypergraph:
    incidence: sparse.csr_matrix
    edge_weights: np.ndarray
    normalized_theta: sparse.csr_matrix
    residual_degrees: np.ndarray
    supports: tuple[tuple[int, ...], ...]
    landmark_ids: tuple[int, ...]
    retained_mass: np.ndarray
    trace_bound: float
    trace_error_exact: float | None


def _checked_ids(landmark_ids: Sequence[int], count: int, n: int) -> tuple[int, ...]:
    try:
        original = tuple(landmark_ids)
        if any(isinstance(value, bool) for value in original):
            raise TypeError("Boolean values are not valid landmark indices")
        ids = tuple(integer_index(value) for value in original)
    except (TypeError, ValueError) as exc:
        raise ValueError("landmark_ids must be integer indices") from exc
    if len(ids) != count or len(set(ids)) != count or any(
        value < 0 or value >= n for value in ids
    ):
        raise ValueError("landmark_ids must be distinct indices in [0, n) and match columns")
    return ids


def _checked_columns(columns: np.ndarray) -> np.ndarray:
    raw = np.asarray(columns)
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] > raw.shape[0]:
        raise ValueError("columns must be an n-by-m matrix with 0 <= m <= n")
    if np.iscomplexobj(raw):
        raise ValueError("transport columns must be real")
    try:
        p = np.asarray(raw, dtype=np.float64).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("transport columns must be numeric") from exc
    if not np.isfinite(p).all() or np.min(p, initial=0.0) < -1e-12:
        raise ValueError("transport columns must be finite and nonnegative")
    # Roundoff from an exponential may leave tiny negative entries or a
    # column sum just off one; remove the former and restore the latter.
    np.maximum(p, 0.0, out=p)
    if p.shape[1]:
        sums = p.sum(axis=0)
        if not np.allclose(sums, 1.0, atol=1e-8, rtol=0.0):
            raise ValueError("each selected transport column must sum to one")
        p /= sums[None, :]
    if np.any(p.sum(axis=1) > 1.0 + 1e-8):
        raise ValueError("selected columns cannot exceed unit mass at any node")
    if p.shape[1] == p.shape[0] and not np.allclose(
        p.sum(axis=1), 1.0, atol=1e-8, rtol=0.0
    ):
        raise ValueError("a complete set of n columns must be doubly stochastic")
    return p


def sparse_transport_hypergraph(
    columns: np.ndarray,
    *,
    top_k: int,
    landmark_ids: Sequence[int] | None = None,
) -> SparseTransportHypergraph:
    """Build a certified sparse incidence matrix from selected endpoint rows.

    ``columns[:, j]`` is one probability vector ``p_r``.  For a real
    symmetric Hamiltonian/Markov generator, a selected column of ``P`` or
    ``C`` equals the corresponding endpoint row by symmetry.  The function
    assumes all selected columns come from one unobserved *doubly stochastic*
    ``n x n`` transport matrix; it validates its observable consequences.

    The largest ``top_k`` positive entries are kept in each column.  Exact
    ties are broken by node index for reproducibility, so hard supports are
    pointwise permutation-equivariant only when no tie crosses the cutoff.
    ``incidence`` always has ``m+n`` columns: ``m`` retained endpoint groups
    followed by ``n`` singleton residual columns.  Zero singleton columns
    are included and must be skipped when forming inverse edge degrees.
    """
    p = _checked_columns(columns)
    n, m = p.shape
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= n:
        raise ValueError("top_k must be an integer in [1, n]")
    if landmark_ids is None:
        ids = tuple(range(m))
    else:
        ids = _checked_ids(landmark_ids, m, n)

    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    supports: list[tuple[int, ...]] = []
    masses = np.zeros(m, dtype=np.float64)
    retained_sq_over_mass = 0.0
    residual = np.ones(n, dtype=np.float64)
    node_indices = np.arange(n)
    for col in range(m):
        # lexsort ranks descending probability and then ascending node id.
        order = np.lexsort((node_indices, -p[:, col]))
        kept = order[:top_k]
        kept = kept[p[kept, col] > 0]
        kept.sort()
        support = tuple(int(i) for i in kept)
        supports.append(support)
        b = p[kept, col]
        mass = float(b.sum())
        masses[col] = mass
        retained_sq_over_mass += float(b @ b) / mass
        residual[kept] -= b
        row_indices.extend(support)
        col_indices.extend([col] * len(support))
        values.extend(float(value) for value in b)

    # Every subtraction above is from a subset of a column-stochastic matrix.
    if np.min(residual) < -1e-8:
        raise ValueError("selected columns leave negative singleton residual mass")
    residual = np.maximum(residual, 0.0)
    node_indices = np.arange(n)
    row_indices.extend(int(i) for i in node_indices if residual[i] > 0)
    col_indices.extend(int(m + i) for i in node_indices if residual[i] > 0)
    values.extend(float(residual[i]) for i in node_indices if residual[i] > 0)
    incidence = sparse.coo_matrix(
        (values, (row_indices, col_indices)), shape=(n, m + n), dtype=np.float64
    ).tocsr()
    selected = incidence[:, :m]
    inv_mass = np.divide(1.0, masses, out=np.zeros_like(masses), where=masses > 0)
    theta = (selected @ sparse.diags(inv_mass) @ selected.T
             + sparse.diags(residual, format="csr")).tocsr()
    theta.eliminate_zeros()

    # tr(K) = sum_r ||p_r||².  Unknown omitted columns sum to the vector
    # u=1−sum_selected p_r.  Cauchy-Schwarz across (n−m) omitted columns
    # gives sum_omitted ||p_r||² >= ||u||²/(n−m).
    trace_theta = retained_sq_over_mass + float(residual.sum())
    trace_selected = float(np.sum(p * p))
    omitted_count = n - m
    if omitted_count:
        missing_mass = np.maximum(1.0 - p.sum(axis=1), 0.0)
        lower_missing_trace = float(missing_mass @ missing_mass) / omitted_count
        trace_bound = max(0.0, trace_theta - trace_selected - lower_missing_trace)
        trace_error_exact = None
    else:
        trace_bound = max(0.0, trace_theta - trace_selected)
        trace_error_exact = trace_bound
    return SparseTransportHypergraph(
        incidence=incidence,
        edge_weights=np.ones(m + n, dtype=np.float64),
        normalized_theta=theta,
        residual_degrees=residual,
        supports=tuple(supports),
        landmark_ids=ids,
        retained_mass=masses,
        trace_bound=trace_bound,
        trace_error_exact=trace_error_exact,
    )


def selected_transport_columns(
    hamiltonian: sparse.spmatrix | np.ndarray,
    times: Iterable[float],
    landmark_ids: Sequence[int],
    *,
    family: str,
    batch_size: int = 16,
) -> list[np.ndarray]:
    """Compute only chosen quantum or matched-Markov endpoint columns.

    ``family`` is ``"quantum"`` or ``"matched_markov"``.  For real
    symmetric ``H``, ``expm_multiply`` on one-hot columns returns the
    measurement-outcome probability vectors needed above without materializing
    the full exponential.  Each returned matrix has shape ``n x m`` in the
    order of ``landmark_ids``.  The same IDs and times should be used for both
    families in a controlled comparison.
    """
    raw = sparse.csr_matrix(hamiltonian)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or raw.shape[0] == 0:
        raise ValueError("Hamiltonian must be a nonempty square matrix")
    if np.iscomplexobj(raw.data):
        raise ValueError("Hamiltonian must be real")
    h = raw.astype(np.float64)
    if not np.isfinite(h.data).all():
        raise ValueError("Hamiltonian must be finite")
    asymmetry = h - h.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1e-12:
        raise ValueError("Hamiltonian must be symmetric")
    if family not in {"quantum", "matched_markov"}:
        raise ValueError("family must be quantum or matched_markov")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    try:
        t = np.asarray(list(times), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("times must be a nonempty sequence") from exc
    if t.ndim != 1 or t.size == 0 or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("times must be finite and nonnegative")
    n = h.shape[0]
    ids = _checked_ids(landmark_ids, len(landmark_ids), n)
    if family == "matched_markov":
        off = h.multiply(h).tocsr()
        off = off - sparse.diags(off.diagonal(), format="csr")
        off.eliminate_zeros()
        q = off - sparse.diags(np.asarray(off.sum(axis=1)).ravel(), format="csr")
    columns_by_time: list[np.ndarray] = []
    for time in t:
        columns = np.empty((n, len(ids)), dtype=np.float64)
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            basis = np.zeros((n, len(batch_ids)), dtype=np.float64)
            basis[list(batch_ids), np.arange(len(batch_ids))] = 1.0
            if family == "quantum":
                action = expm_multiply((-1j * float(time)) * h, basis)
                columns[:, start:start + len(batch_ids)] = np.abs(action) ** 2
            else:
                action = expm_multiply((float(time) ** 2) * q, basis)
                columns[:, start:start + len(batch_ids)] = np.asarray(action).real
        columns_by_time.append(columns)
    return columns_by_time
