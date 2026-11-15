"""Fixed-decoder evaluation and direct operator-equivalence oracles."""

import numpy as np
import pytest
from scipy import sparse

from qhyper.construction_evaluation import (
    assert_operator_equivalence,
    evaluate_fusion,
    evaluate_operator,
    fused_operator,
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


def test_fusion_endpoints_and_train_only_selection():
    graph, features, labels, split = _sample()
    graph = sparse.csr_matrix(graph)
    identity = sparse.eye(graph.shape[0], format="csr")
    np.testing.assert_array_equal(fused_operator(graph, identity, 0.0).toarray(),
                                  graph.toarray())
    np.testing.assert_array_equal(fused_operator(graph, identity, 1.0).toarray(),
                                  identity.toarray())
    assert sparse.issparse(fused_operator(graph, identity, 0.5))

    graph_only = evaluate_operator(graph, features, labels, split, C_grid=(0.1, 1.0))
    endpoint = evaluate_fusion(graph, identity, features, labels, split,
                               alphas=(0.0,), C_grid=(0.1, 1.0))
    assert endpoint["selected_alpha"] == 0.0
    for key, value in graph_only.items():
        assert endpoint[key] == value

    reference = evaluate_fusion(graph, identity, features, labels, split,
                                alphas=(0.0, 0.5, 1.0), C_grid=(0.1, 1.0))
    permuted_labels = labels.copy()
    permuted_labels[split["test"]] = np.roll(labels[split["test"]], 3)
    permuted = evaluate_fusion(graph, identity, features, permuted_labels, split,
                               alphas=(0.0, 0.5, 1.0), C_grid=(0.1, 1.0))
    for key in ("selected_alpha", "selected_C", "validation_by_alpha",
                "validation_log_loss", "validation_macro_f1", "model_fingerprint"):
        assert reference[key] == permuted[key]


def test_fusion_rejects_invalid_weights_and_shape():
    graph = sparse.eye(3, format="csr")
    with pytest.raises(ValueError, match="alpha"):
        fused_operator(graph, graph, 1.1)
    with pytest.raises(ValueError, match="same shape"):
        fused_operator(graph, sparse.eye(4, format="csr"), 0.5)
