"""Independent construction-task data must have auditable latent causes."""

import numpy as np
import pytest

from qhyper.construction_synthetic import make_synthetic_graph, stratified_node_split


@pytest.mark.parametrize("regime", ["group", "pairwise", "heterophily"])
def test_generator_is_reproducible_binary_and_exogenous(regime):
    first = make_synthetic_graph(regime, seed=17, n=60)
    repeated = make_synthetic_graph(regime, seed=17, n=60)
    graph, features, labels, metadata = first
    assert (graph != repeated[0]).nnz == 0
    np.testing.assert_array_equal(features, repeated[1])
    np.testing.assert_array_equal(labels, repeated[2])
    assert metadata == repeated[3]
    assert graph.shape == (60, 60)
    assert (graph != graph.T).nnz == 0
    assert graph.diagonal().sum() == 0
    assert set(graph.data) <= {1}
    assert metadata["realized_edges"] == graph.nnz // 2
    assert metadata["regime"] == regime
    assert len(np.unique(labels)) == 3


def test_regimes_share_class_and_feature_draw_but_change_edge_mechanism():
    samples = {name: make_synthetic_graph(name, seed=21) for name in
               ("group", "pairwise", "heterophily")}
    for name in ("pairwise", "heterophily"):
        np.testing.assert_array_equal(samples["group"][1], samples[name][1])
        np.testing.assert_array_equal(samples["group"][2], samples[name][2])
        assert (samples["group"][0] != samples[name][0]).nnz > 0
    graph, _, labels, metadata = samples["group"]
    assert metadata["latent_groups_diagnostic_only"]
    for group in metadata["latent_groups_diagnostic_only"]:
        assert len(group) == 4
        assert len({int(labels[node]) for node in group}) == 1
        for offset, u in enumerate(group):
            for v in group[offset + 1:]:
                assert graph[u, v] == 1
    assert samples["pairwise"][3]["latent_groups_diagnostic_only"] == []
    assert samples["heterophily"][3]["latent_groups_diagnostic_only"] == []
    # Conditional fixed-count nulls remove total density as a confound.
    assert len({sample[0].nnz for sample in samples.values()}) == 1
    group_within = [
        sum(int(graph[u, v]) for u in np.flatnonzero(labels == k)
            for v in np.flatnonzero(labels == k) if u < v)
        for k in range(3)
    ]
    pairwise_graph = samples["pairwise"][0]
    pairwise_within = [
        sum(int(pairwise_graph[u, v]) for u in np.flatnonzero(labels == k)
            for v in np.flatnonzero(labels == k) if u < v)
        for k in range(3)
    ]
    assert group_within == pairwise_within


def test_heterophily_has_larger_cross_class_pair_probability():
    _, _, _, metadata = make_synthetic_graph("heterophily", seed=5)
    parameters = metadata["parameters"]
    assert parameters["heterophily_cross"] > parameters["heterophily_within"]


def test_stratified_split_covers_nodes_without_overlap():
    _, _, labels, _ = make_synthetic_graph("group", seed=5)
    first = stratified_node_split(labels, seed=90)
    repeated = stratified_node_split(labels, seed=90)
    indices = np.concatenate([first[role] for role in ("train", "validation", "test")])
    np.testing.assert_array_equal(np.sort(indices), np.arange(len(labels)))
    for role in first:
        np.testing.assert_array_equal(first[role], repeated[role])
        assert set(np.unique(labels[first[role]])) == {0, 1, 2}


def test_invalid_generator_and_split_arguments_are_rejected():
    with pytest.raises(ValueError, match="regime"):
        make_synthetic_graph("unknown", 1)
    with pytest.raises(ValueError, match="at least 12"):
        make_synthetic_graph("group", 1, n=11)
    with pytest.raises(ValueError, match="at least three"):
        stratified_node_split(np.array([0, 0, 1, 1, 1]), seed=1)
