"""Candidate-subspace spectral features for a falsifiable P1 revision.

All quantities here are deterministic functions of the observed graph's
Hamiltonian.  They are not observations of physical many-body interactions.
The original P1 variant names are deliberately untouched: the new columns are
called Qvol, Qcorr and Qspec, while Cgram is their analytic short-time control.

For the current small real graphs (143 and 327 nodes), one eigendecomposition
per graph is shared by all candidates and times.  The quantum function falls
back to sparse actions on larger graphs; matched heat/wave features currently
require the shared dense eigensystem and explicitly reject oversized inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from .group_response import canonical_candidate
from .quantum_action import checked_hamiltonian


_SPECTRAL_COLUMNS = ("Qvol_log_loss", "Qcorr_log_det", "Qspec_trace_g2_per_node",
                     "Qspec_trace_g3_per_node")
_Q3_COLUMNS = ("Q3_R_coh", "Q3_R_inc", "Q3_C")
_CLASSICAL_COLUMNS = ("Cgram_boundary_energy_per_node", "Cgram_offdiag_sq",
                      "Cgram_trace_jtj2_per_node", "Cgram_trace_jtj3_per_node")


def _times(times: Sequence[float] | np.ndarray, *, nonnegative: bool = False) -> np.ndarray:
    grid = np.asarray(times, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all():
        raise ValueError("times must be a nonempty finite one-dimensional sequence")
    if np.unique(grid).size != grid.size:
        raise ValueError("times must be distinct so feature names remain unique")
    if nonnegative and np.any(grid < 0):
        raise ValueError("heat and wave times must be nonnegative")
    return grid


def _candidate_arrays(candidates: Iterable[Iterable[int]], n: int) -> list[np.ndarray]:
    return [np.asarray(canonical_candidate(candidate, n), dtype=np.int64)
            for candidate in candidates]


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


def _unitary_group_block(operator: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return internal retention and *external leakage* Gram matrices.

    The complement matrix is computed independently so tests and callers can
    check the exact unitary identity M + G = I.  For nonunitary classical
    operators, use :func:`group_retention` instead; I-M is then a deficit, not
    an external probability leakage matrix.
    """
    n = operator.shape[0]
    inside = operator[np.ix_(nodes, nodes)]
    outside_mask = np.ones(n, dtype=bool)
    outside_mask[nodes] = False
    outside = operator[np.ix_(np.flatnonzero(outside_mask), nodes)]
    M = _hermitian(inside.conj().T @ inside)
    G = _hermitian(outside.conj().T @ outside)
    return M, G


def group_gram_from_operator(
    operator: np.ndarray, candidate: Iterable[int], *, check_unitary: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``M=U[e,e]^*U[e,e]`` and ``G=U[e^c,e]^*U[e^c,e]``.

    ``check_unitary=True`` guards the interpretation of G as probability
    leakage.  The return order is ``(M, G)``.  Floating residuals at 1e-10
    scale are tolerated; a nonunitary operator is rejected.
    """
    U = np.asarray(operator, dtype=np.complex128)
    if U.ndim != 2 or U.shape[0] != U.shape[1] or not np.isfinite(U).all():
        raise ValueError("operator must be a finite square matrix")
    nodes = np.asarray(canonical_candidate(candidate, U.shape[0]), dtype=np.int64)
    if check_unitary and not np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-9):
        raise ValueError("operator must be unitary for an external-leakage Gram")
    return _unitary_group_block(U, nodes)


def group_retention(operator: np.ndarray, candidate: Iterable[int]) -> np.ndarray:
    """Return ``operator[e,e]^*operator[e,e]`` without a unitary interpretation."""
    K = np.asarray(operator, dtype=np.complex128)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or not np.isfinite(K).all():
        raise ValueError("operator must be a finite square matrix")
    nodes = np.asarray(canonical_candidate(candidate, K.shape[0]), dtype=np.int64)
    block = K[np.ix_(nodes, nodes)]
    return _hermitian(block.conj().T @ block)


def q3_from_retention(M: np.ndarray) -> tuple[float, float, float]:
    """Recover old ``(R_coh, R_inc, C)`` from a candidate retention matrix."""
    value = np.asarray(M, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 2:
        raise ValueError("M must be a square candidate matrix of size >= 2")
    k = value.shape[0]
    coherent = float(np.real(np.sum(value)) / k)
    incoherent = float(np.real(np.trace(value)) / k)
    return coherent, incoherent, coherent - incoherent


def operator_spectral_statistics(M: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Return Qvol, Qcorr, and two deficit-Gram spectral moments.

    The regularized retention is ``(1-eps) M + eps I``.  ``M`` must be a PSD
    contraction, as it is for a unitary subblock or a contraction kernel.  The
    returned values are nonnegative up to floating error.  For a nonunitary
    classical kernel, ``I-M`` is *not* an external-leakage probability Gram.
    """
    matrix = np.asarray(M, dtype=np.complex128)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or
            matrix.shape[0] < 2 or not np.isfinite(matrix).all()):
        raise ValueError("M must be a finite square matrix of size >= 2")
    if not np.isfinite(eps) or not 0 < eps < 1:
        raise ValueError("eps must lie strictly between 0 and 1")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-8):
        raise ValueError("M must be Hermitian")
    matrix = _hermitian(matrix)
    eig = np.linalg.eigvalsh(matrix)
    if eig.min() < -1e-8 or eig.max() > 1 + 1e-8:
        raise ValueError("M must be a PSD contraction")
    k = matrix.shape[0]
    regularized = _hermitian((1 - eps) * matrix + eps * np.eye(k))
    chol = np.linalg.cholesky(regularized)
    logdet = float(2 * np.log(np.real(np.diag(chol))).sum())
    diagonal = np.real(np.diag(regularized))
    omega = float(np.log(diagonal).sum() - logdet)
    if omega < -1e-8:
        raise ArithmeticError("Hadamard non-additivity is materially negative")
    deficit = _hermitian(np.eye(k) - matrix)
    trace2 = float(np.real(np.trace(deficit @ deficit)) / k)
    trace3 = float(np.real(np.trace(deficit @ deficit @ deficit)) / k)
    return np.array([-logdet / k, omega, trace2, trace3], dtype=np.float64)


def _feature_names(times: np.ndarray, prefix: str) -> tuple[str, ...]:
    return tuple(f"{column if prefix == 'Q' else prefix + '_' + column}_t{float(time):g}"
                 for time in times for column in _SPECTRAL_COLUMNS)


def _q3_names(times: np.ndarray) -> tuple[str, ...]:
    return tuple(f"{column}_t{float(time):g}" for time in times for column in _Q3_COLUMNS)


def _dense_eigensystem(h: sparse.csr_matrix, max_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    if h.shape[0] > max_nodes:
        raise ValueError(f"shared eigendecomposition requires n <= {max_nodes}; "
                         "use sparse-action quantum extraction for larger graphs")
    values, vectors = np.linalg.eigh(h.toarray())
    return values, vectors


def _operator(vectors: np.ndarray, factors: np.ndarray) -> np.ndarray:
    return (vectors * factors[np.newaxis, :]) @ vectors.T


def _block_from_action(h: sparse.csr_matrix, nodes: np.ndarray, time: float) -> np.ndarray:
    n = h.shape[0]
    if time == 0:
        return np.eye(nodes.size, dtype=np.complex128)
    states = np.zeros((n, nodes.size), dtype=np.complex128)
    states[nodes, np.arange(nodes.size)] = 1.0
    evolved = np.asarray(expm_multiply((-1j * time) * h, states))
    return evolved[nodes, :]


def quantum_group_feature_matrices(
    hamiltonian: sparse.spmatrix | np.ndarray,
    candidates: Iterable[Iterable[int]],
    times: Sequence[float] | np.ndarray,
    *,
    eps: float = 1e-6,
    dense_limit: int = 400,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    """Extract old Q3 and new Qvol/Qcorr/Qspec from the *same* subblocks.

    Q3 retains its original ``[R_coh,R_inc,C]`` column order at each time;
    Qspec has ``[L_vol,Omega,TrG²/k,TrG³/k]``.  For ``n<=dense_limit`` a single
    real symmetric eigendecomposition is reused for every candidate/time;
    larger graphs use bounded sparse actions per candidate.  This is a
    computational optimization, not additional structural information.
    """
    h = checked_hamiltonian(hamiltonian)
    grid = _times(times)
    groups = _candidate_arrays(candidates, h.shape[0])
    values = np.zeros((len(groups), 4 * len(grid)), dtype=np.float64)
    q3 = np.zeros((len(groups), 3 * len(grid)), dtype=np.float64)
    if h.shape[0] <= dense_limit:
        eig, vec = _dense_eigensystem(h, dense_limit)
        for time_index, time in enumerate(grid):
            U = _operator(vec, np.exp(-1j * float(time) * eig))
            for group_index, nodes in enumerate(groups):
                block = U[np.ix_(nodes, nodes)]
                M = _hermitian(block.conj().T @ block)
                values[group_index, 4 * time_index:4 * (time_index + 1)] = (
                    operator_spectral_statistics(M, eps=eps))
                q3[group_index, 3 * time_index:3 * (time_index + 1)] = (
                    q3_from_retention(M))
    else:
        for time_index, time in enumerate(grid):
            for group_index, nodes in enumerate(groups):
                block = _block_from_action(h, nodes, float(time))
                M = _hermitian(block.conj().T @ block)
                values[group_index, 4 * time_index:4 * (time_index + 1)] = (
                    operator_spectral_statistics(M, eps=eps))
                q3[group_index, 3 * time_index:3 * (time_index + 1)] = (
                    q3_from_retention(M))
    return ({"Q3": q3, "Qspec": values},
            {"Q3": _q3_names(grid), "Qspec": _feature_names(grid, "Q")})


def quantum_spectral_feature_matrix(
    hamiltonian: sparse.spmatrix | np.ndarray,
    candidates: Iterable[Iterable[int]],
    times: Sequence[float] | np.ndarray,
    *,
    eps: float = 1e-6,
    dense_limit: int = 400,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Convenience wrapper returning just the Qvol/Qcorr/Qspec columns."""
    matrices, names = quantum_group_feature_matrices(
        hamiltonian, candidates, times, eps=eps, dense_limit=dense_limit)
    return matrices["Qspec"], names["Qspec"]


def classical_boundary_gram_matrix(
    hamiltonian: sparse.spmatrix | np.ndarray,
    candidates: Iterable[Iterable[int]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Cheap exact controls for the small-time Qvol/Qcorr/Qspec coefficients.

    ``J=H[e^c,e]``.  The Qvol leading coefficient is ``(1-eps)`` times
    column 0, while Qcorr's is ``(1-eps)^2`` times column 1.  These columns
    use only the observed graph and must accompany any Qvol/Qcorr claim.
    """
    h = checked_hamiltonian(hamiltonian)
    groups = _candidate_arrays(candidates, h.shape[0])
    result = np.zeros((len(groups), len(_CLASSICAL_COLUMNS)), dtype=np.float64)
    for index, nodes in enumerate(groups):
        outside = np.ones(h.shape[0], dtype=bool)
        outside[nodes] = False
        J = h[outside, :][:, nodes].toarray()
        gram = J.T @ J
        k = nodes.size
        result[index] = [
            float(np.trace(gram) / k),
            float(np.sum(np.abs(np.triu(gram, 1)) ** 2)),
            float(np.trace(gram @ gram) / k),
            float(np.trace(gram @ gram @ gram) / k),
        ]
    return result, _CLASSICAL_COLUMNS


def matched_classical_spectral_feature_matrix(
    hamiltonian: sparse.spmatrix | np.ndarray,
    candidates: Iterable[Iterable[int]],
    times: Sequence[float] | np.ndarray,
    *,
    kind: str,
    eps: float = 1e-6,
    laplacian_factor: float = 2.0,
    dense_limit: int = 400,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Apply the *same* group functional to heat or classical-wave operators.

    For the primary protocol ``H=L/2`` and ``laplacian_factor=2`` restores
    ``L``.  Heat uses ``exp(-tL)`` and wave uses ``cos(t sqrt(L))``.  Both are
    contractions for PSD L.  Their Gram deficit ``I-K_ee^*K_ee`` includes
    nonunitary attenuation and must not be called external probability leak.
    The pilot graphs fit the shared dense eigensystem; larger graphs require
    a separate action-based implementation before this control can run.
    """
    if kind not in {"heat", "wave"}:
        raise ValueError("kind must be 'heat' or 'wave'")
    if not np.isfinite(laplacian_factor) or laplacian_factor <= 0:
        raise ValueError("laplacian_factor must be positive and finite")
    h = checked_hamiltonian(hamiltonian)
    grid = _times(times, nonnegative=True)
    groups = _candidate_arrays(candidates, h.shape[0])
    eig, vec = _dense_eigensystem(h, dense_limit)
    lap_eig = eig * laplacian_factor
    if lap_eig.min() < -1e-8:
        raise ValueError("matched heat/wave kernels require PSD Laplacian")
    lap_eig = np.maximum(lap_eig, 0.0)
    result = np.zeros((len(groups), 4 * len(grid)), dtype=np.float64)
    for time_index, time in enumerate(grid):
        if kind == "heat":
            factors = np.exp(-float(time) * lap_eig)
        else:
            factors = np.cos(float(time) * np.sqrt(lap_eig))
        K = _operator(vec, factors)
        for group_index, nodes in enumerate(groups):
            block = K[np.ix_(nodes, nodes)]
            M = _hermitian(block.conj().T @ block)
            result[group_index, 4 * time_index:4 * (time_index + 1)] = (
                operator_spectral_statistics(M, eps=eps))
    return result, _feature_names(grid, "Cmatched_" + kind)
