"""Memory-bounded continuous-time quantum-walk actions.

Production code applies the sparse exponential to batches of initial states.
The dense unitary is available only as a small-graph numerical oracle.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply


def checked_hamiltonian(hamiltonian: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    """Validate the protocol's real symmetric Hamiltonian and return CSR."""
    raw = sparse.csr_matrix(hamiltonian)
    if np.iscomplexobj(raw.data) and np.any(np.imag(raw.data) != 0):
        raise ValueError("Hamiltonian must be real symmetric")
    h = (raw.real if np.iscomplexobj(raw.data) else raw).astype(np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] == 0:
        raise ValueError("Hamiltonian must be a nonempty square matrix")
    h.sum_duplicates()
    h.eliminate_zeros()
    if not np.isfinite(h.data).all():
        raise ValueError("Hamiltonian contains nonfinite values")
    asymmetry = h - h.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1e-10:
        raise ValueError("Hamiltonian must be real symmetric")
    h = ((h + h.T) * 0.5).tocsr()
    h.eliminate_zeros()
    return h


def _checked_states(states: np.ndarray, n_nodes: int) -> np.ndarray:
    s = np.asarray(states, dtype=np.complex128)
    if s.ndim == 1:
        s = s[:, None]
    if s.ndim != 2 or s.shape[0] != n_nodes:
        raise ValueError("initial_states must have shape (N, batch)")
    if not np.isfinite(s).all():
        raise ValueError("initial_states contain nonfinite values")
    return s


def _checked_times(times: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("times must be a nonempty finite one-dimensional sequence")
    return values


def quantum_action(
    hamiltonian: sparse.spmatrix | np.ndarray,
    initial_states: np.ndarray,
    times: Sequence[float] | np.ndarray,
    *,
    batch_size: int = 64,
) -> Iterator[tuple[float, slice, np.ndarray]]:
    """Yield ``(time, columns, exp(-itH) @ initial_states[:, columns])``.

    Batches are outermost so at most ``N * batch_size`` complex amplitudes
    reside in an evolved-state buffer.  The full dense propagator is never
    materialized.  The caller may consume each result immediately to keep
    memory bounded across times and candidates.
    """
    h = checked_hamiltonian(hamiltonian)
    states = _checked_states(initial_states, h.shape[0])
    grid = _checked_times(times)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, states.shape[1], batch_size):
        columns = slice(start, min(start + batch_size, states.shape[1]))
        block = states[:, columns]
        for time in grid:
            if time == 0.0:
                evolved = block.copy()
            else:
                evolved = np.asarray(expm_multiply((-1j * time) * h, block))
                if evolved.ndim == 1:
                    evolved = evolved[:, None]
            yield float(time), columns, evolved


def reference_unitary(
    hamiltonian: sparse.spmatrix | np.ndarray,
    time: float,
    *,
    max_nodes: int = 512,
) -> np.ndarray:
    """Compute a dense small-graph unitary for tests, never production runs."""
    h = checked_hamiltonian(hamiltonian)
    if h.shape[0] > max_nodes:
        raise ValueError("dense reference exceeds max_nodes")
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    return np.asarray(expm((-1j * float(time)) * h.toarray()))


def reference_dense_action(
    hamiltonian: sparse.spmatrix | np.ndarray,
    initial_states: np.ndarray,
    times: Sequence[float] | np.ndarray,
    *,
    max_nodes: int = 512,
) -> dict[float, np.ndarray]:
    """Return dense-oracle actions keyed by time for small numerical tests."""
    h = checked_hamiltonian(hamiltonian)
    states = _checked_states(initial_states, h.shape[0])
    grid = _checked_times(times)
    return {
        float(t): reference_unitary(h, float(t), max_nodes=max_nodes) @ states
        for t in grid
    }
