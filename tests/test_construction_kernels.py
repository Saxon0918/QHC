"""Mathematical oracles for the small-graph construction kernels."""

import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import expm

from qhyper.construction_kernels import (
    matched_markov,
    quantum_transport,
    spread_matched_markov,
)
from qhyper.hamiltonian import build_hamiltonian


def _path_hamiltonian() -> sparse.csr_matrix:
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=float,
    )
    return build_hamiltonian(adjacency)


def _hgnn_operator(incidence: np.ndarray, weights: np.ndarray) -> np.ndarray:
    edge_degree = incidence.sum(axis=0)
    node_degree = incidence @ weights
    middle = (incidence * (weights / edge_degree)[None, :]) @ incidence.T
    inv_root = 1.0 / np.sqrt(node_degree)
    return inv_root[:, None] * middle * inv_root[None, :]


def test_quantum_transport_is_bistochastic_psd_and_exact_soft_hgnn():
    h = _path_hamiltonian()
    result = quantum_transport(h, [0.2, 1.3], weights=[1, 3])
    p0, p1 = result["transport"]
    weights = result["weights"]
    np.testing.assert_allclose(weights, [0.25, 0.75])
    for p in (p0, p1):
        assert np.min(p) >= 0
        np.testing.assert_allclose(p, p.T, atol=1e-13)
        np.testing.assert_allclose(p.sum(axis=0), 1, atol=1e-13)
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-13)
    kernel = result["K"]
    np.testing.assert_allclose(kernel, 0.25 * p0.T @ p0 + 0.75 * p1.T @ p1)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-13)
    np.testing.assert_allclose(kernel.sum(axis=0), 1, atol=1e-13)
    assert np.linalg.eigvalsh(kernel).min() >= -1e-13
    assert np.linalg.eigvalsh(kernel).max() <= 1 + 1e-13

    # One soft hyperedge per measured endpoint and time.  Its incidence is
    # P_t.T, while its scalar weight is the corresponding time weight.
    incidence = np.concatenate((p0.T, p1.T), axis=1)
    edge_weights = np.repeat(weights, h.shape[0])
    np.testing.assert_allclose(_hgnn_operator(incidence, edge_weights), kernel,
                               atol=1e-13)


def test_matched_markov_is_markov_and_has_exact_heat_gram():
    h = _path_hamiltonian()
    time = 0.63
    result = matched_markov(h, [time])
    q = result["generator"]
    c = result["transport"][0]
    np.testing.assert_allclose(q, q.T)
    np.testing.assert_allclose(q.sum(axis=1), 0, atol=1e-15)
    assert np.min(q - np.diag(np.diag(q))) >= 0
    assert np.min(c) >= -1e-14
    np.testing.assert_allclose(c.sum(axis=1), 1, atol=1e-13)
    np.testing.assert_allclose(result["K"], expm(2 * time * time * q),
                               atol=1e-13)


def test_zero_time_and_disconnected_components():
    h = np.array([[0.5, -0.5, 0], [-0.5, 0.5, 0], [0, 0, 0]])
    for builder in (quantum_transport, matched_markov):
        zero = builder(h, [0])
        np.testing.assert_allclose(zero["transport"][0], np.eye(3))
        np.testing.assert_allclose(zero["K"], np.eye(3))
        evolved = builder(h, [0.7])
        np.testing.assert_allclose(evolved["transport"][0][2], [0, 0, 1])
        np.testing.assert_allclose(evolved["K"][2], [0, 0, 1])


def test_permutation_equivariance():
    h = _path_hamiltonian().toarray()
    permutation = [2, 0, 3, 1]
    for builder in (quantum_transport, matched_markov):
        original = builder(h, [0.2, 0.9], weights=[2, 1])
        relabelled = builder(h[np.ix_(permutation, permutation)],
                             [0.2, 0.9], weights=[2, 1])
        np.testing.assert_allclose(relabelled["K"],
                                   original["K"][np.ix_(permutation, permutation)],
                                   atol=1e-13)
        for p, p_relabelled in zip(original["transport"], relabelled["transport"]):
            np.testing.assert_allclose(p_relabelled,
                                       p[np.ix_(permutation, permutation)],
                                       atol=1e-13)


def test_short_time_matches_classical_through_quadratic_order():
    h = _path_hamiltonian()
    errors = []
    for time in (0.3, 0.15):
        quantum = quantum_transport(h, [time])
        classical = matched_markov(h, [time])
        q = classical["generator"]
        np.testing.assert_allclose(
            (quantum["transport"][0] - np.eye(h.shape[0])) / time**2,
            q,
            atol=0.01,
        )
        errors.append(np.linalg.norm(quantum["K"] - classical["K"], ord="fro"))
    # The first unmatched term is fourth order: halving t reduces the error
    # by roughly 16 (up to higher-order corrections).
    assert errors[0] > 10 * errors[1] > 0


def test_spread_matched_markov_matches_each_quantum_trace():
    h = _path_hamiltonian()
    times = [0.2, 0.7, 1.5]
    result = spread_matched_markov(h, times, weights=[1, 2, 1])
    quantum = quantum_transport(h, times)
    np.testing.assert_allclose(result["weights"], [0.25, 0.5, 0.25])
    for index, classical in enumerate(result["transport"]):
        target = float(np.sum(quantum["transport"][index] ** 2))
        assert result["markov_times"][index] >= 0
        assert np.isfinite(result["markov_times"][index])
        np.testing.assert_allclose(classical.sum(axis=1), 1, atol=1e-12)
        np.testing.assert_allclose(classical, classical.T, atol=1e-12)
        assert result["target_traces"][index] == pytest.approx(target,
                                                                abs=1e-10)
        assert result["achieved_traces"][index] == pytest.approx(target,
                                                                  abs=1e-10)
    expected = sum(weight * matrix.T @ matrix for weight, matrix in zip(
        result["weights"], result["transport"]))
    np.testing.assert_allclose(result["K"], expected, atol=1e-13)


def test_spread_match_has_same_short_time_scale_as_quadratic_control():
    h = _path_hamiltonian()
    time = 0.04
    result = spread_matched_markov(h, [time])
    assert result["markov_times"][0] / time**2 == pytest.approx(1.0, rel=0.01)


def test_spread_match_handles_zero_and_uniform_limiting_distributions():
    # At pi/4 a two-vertex quantum walk is exactly uniform; no finite
    # classical Markov time attains its trace-one limiting kernel.
    h = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = spread_matched_markov(h, [0.0, np.pi / 4])
    np.testing.assert_allclose(result["transport"][0], np.eye(2))
    assert result["markov_times"][0] == 0.0
    assert np.isinf(result["markov_times"][1])
    np.testing.assert_allclose(result["transport"][1], np.full((2, 2), 0.5))
    np.testing.assert_allclose(result["target_traces"], [2.0, 1.0], atol=1e-12)

    # The isolated third vertex contributes one additional stationary mode.
    disconnected = np.pad(h, ((0, 1), (0, 1)))
    parts = spread_matched_markov(disconnected, [np.pi / 4])
    assert parts["connected_components"] == 2
    assert np.isinf(parts["markov_times"][0])
    np.testing.assert_allclose(parts["transport"][0],
                               np.array([[0.5, 0.5, 0],
                                         [0.5, 0.5, 0],
                                         [0, 0, 1]]))

    # A diagonal Hamiltonian induces no mixing under either transport.
    diagonal = spread_matched_markov(np.diag([0.0, 1.0, 3.0]), [0.0, 4.0])
    assert diagonal["connected_components"] == 3
    np.testing.assert_allclose(diagonal["markov_times"], [0.0, 0.0])
    for classical in diagonal["transport"]:
        np.testing.assert_allclose(classical, np.eye(3))


def test_spread_match_permutation_equivariance():
    h = _path_hamiltonian().toarray()
    permutation = [2, 0, 3, 1]
    original = spread_matched_markov(h, [0.3, 0.9])
    permuted = spread_matched_markov(h[np.ix_(permutation, permutation)],
                                    [0.3, 0.9])
    np.testing.assert_allclose(permuted["markov_times"],
                               original["markov_times"], atol=1e-12)
    np.testing.assert_allclose(permuted["K"],
                               original["K"][np.ix_(permutation, permutation)],
                               atol=1e-12)


@pytest.mark.parametrize(
    "hamiltonian,times,weights,max_nodes,match",
    [
        (np.ones((2, 3)), [1], None, 128, "square"),
        (np.eye(3), [1], None, 2, "max_nodes"),
        (np.array([[0, 1], [0, 0]]), [1], None, 128, "symmetric"),
        (1j * np.eye(2), [1], None, 128, "real"),
        (np.eye(2), [-1], None, 128, "nonnegative"),
        (np.eye(2), [np.nan], None, 128, "finite"),
        (np.eye(2), [1, 2], [1], 128, "match times"),
        (np.eye(2), [1], [0], 128, "positive sum"),
    ],
)
def test_invalid_inputs_are_rejected(hamiltonian, times, weights, max_nodes, match):
    for builder in (quantum_transport, matched_markov, spread_matched_markov):
        with pytest.raises(ValueError, match=match):
            builder(hamiltonian, times, weights=weights, max_nodes=max_nodes)
