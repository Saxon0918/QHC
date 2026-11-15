"""Fixed-decoder evaluation and direct operator-equivalence oracles."""

import numpy as np
import pytest
from scipy import sparse

from qhyper.construction_evaluation import (
    assert_operator_equivalence,
    evaluate_operator,
    operator_equivalence_error,
)
from qhyper.construction_synthetic import make_synthetic_graph, stratified_node_split


def _sample():
    graph, features, labels, _ = make_synthetic_graph("group", seed=33)
    split = stratified_node_split(labels, seed=13)
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    theta = np.diag(1.0 / np.maximum(degrees, 1.0)) @ graph.toarray()
    return theta, features, labels, split


def test_test_label_permutation_does_not_change_fitting_or_c_selection():
    theta, features, labels, split = _sample()
    reference = evaluate_operator(theta, features, labels, split, C_grid=(0.1, 1.0, 10.0))
    permuted = labels.copy()
    permuted[split["test"]] = np.roll(labels[split["test"]], 3)
    changed = evaluate_operator(theta, features, permuted, split, C_grid=(0.1, 1.0, 10.0))
    assert reference["selected_C"] == changed["selected_C"]
    assert reference["validation_log_loss"] == changed["validation_log_loss"]
    assert reference["validation_macro_f1"] == changed["validation_macro_f1"]
    assert reference["model_fingerprint"] == changed["model_fingerprint"]
    assert 0.0 <= reference["test_macro_f1"] <= 1.0
    assert np.isfinite(reference["test_log_loss"])


def test_identical_hypergraph_and_graph_operators_have_identical_decoder_results():
    _, features, labels, split = _sample()
    # Column-stochastic incidence with unit node/hyperedge degrees: the
    # one-layer linear HGNN operator is directly the weighted graph K=B B^T.
    incidence = np.array([
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
    ])
    full_incidence = np.eye(len(labels))
    full_incidence[:3, :3] = incidence
    small_operator = incidence @ incidence.T
    direct_graph = np.eye(len(labels))
    direct_graph[:3, :3] = small_operator
    hypergraph_operator = full_incidence @ full_incidence.T
    error = assert_operator_equivalence(direct_graph, hypergraph_operator)
    assert error == operator_equivalence_error(direct_graph, hypergraph_operator) == 0.0
    np.testing.assert_array_equal(direct_graph @ features, hypergraph_operator @ features)
    first = evaluate_operator(direct_graph, features, labels, split)
    second = evaluate_operator(hypergraph_operator, features, labels, split)
    assert first == second


def test_operator_equivalence_fails_for_distinct_operators():
    identity = np.eye(3)
    changed = identity.copy()
    changed[0, 1] = 0.01
    assert operator_equivalence_error(identity, changed) == pytest.approx(0.01)
    with pytest.raises(AssertionError, match="differ"):
        assert_operator_equivalence(identity, changed)


def test_evaluator_rejects_overlapping_splits_and_invalid_c_grid():
    theta, features, labels, split = _sample()
    invalid_split = {name: indices.copy() for name, indices in split.items()}
    invalid_split["test"][0] = invalid_split["train"][0]
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_operator(theta, features, labels, invalid_split)
    with pytest.raises(ValueError, match="C_grid"):
        evaluate_operator(theta, features, labels, split, C_grid=(0.0,))


def test_large_sparse_operator_never_calls_toarray():
    n = 1000
    labels = np.arange(n) % 2
    rng = np.random.default_rng(99)
    features = rng.normal(size=(n, 2))
    features[:, 0] += 0.6 * labels
    split = stratified_node_split(labels, seed=55)
    class NoDenseCSR(sparse.csr_matrix):
        def toarray(self, *args, **kwargs):
            raise AssertionError("CSR propagation operator was densified")

    operator = NoDenseCSR(sparse.eye(n, format="csr"))
    result = evaluate_operator(operator, features, labels, split, C_grid=(1.0,))
    assert 0.0 <= result["test_macro_f1"] <= 1.0
    assert np.isfinite(result["test_log_loss"])
    assert_operator_equivalence(operator, operator.copy())
    changed = NoDenseCSR(operator + sparse.csr_matrix(
        ([0.1], ([0], [1])), shape=(n, n)
    ))
    assert operator_equivalence_error(operator, changed) == pytest.approx(0.1)


def test_sparse_operator_validation_rejects_nonfinite_and_complex():
    _, features, labels, split = _sample()
    n = len(labels)
    nonfinite = sparse.eye(n, format="csr")
    nonfinite.data[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_operator(nonfinite, features, labels, split)
    complex_operator = sparse.eye(n, dtype=np.complex128, format="csr")
    with pytest.raises(ValueError, match="real"):
        evaluate_operator(complex_operator, features, labels, split)
