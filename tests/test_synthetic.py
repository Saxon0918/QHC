import numpy as np

from qhyper.synthetic import (SyntheticConfig, generate_synthetic,
                              triangle_ambiguity_example)
from qhyper.projection import project_hypergraph


def test_synthetic_seed_reproducibility_and_projection():
    config = SyntheticConfig(case="S1", n_nodes=60, num_hyperedges=40,
                             include_multiplicity=True)
    a = generate_synthetic(config, seed=17)
    b = generate_synthetic(config, seed=17)
    c = generate_synthetic(config, seed=18)
    assert a.graph.hyperedges_truth == b.graph.hyperedges_truth
    assert (a.graph.adjacency_csr != b.graph.adjacency_csr).nnz == 0
    assert a.graph.hyperedges_truth != c.graph.hyperedges_truth
    expected_A = project_hypergraph(a.graph.hyperedges_truth, 60, "binary")
    expected_W = project_hypergraph(a.graph.hyperedges_truth, 60, "multiplicity")
    assert (a.graph.adjacency_csr != expected_A).nnz == 0
    assert (a.multiplicity_csr != expected_W).nnz == 0
    assert len(a.graph.hyperedges_truth) == config.num_hyperedges
    assert a.parameters["decoy_triangle_count"] == 5


def test_s2_generates_high_overlap_cliques():
    sample = generate_synthetic(SyntheticConfig(case="S2", n_nodes=60,
                                                num_hyperedges=40), seed=17)
    degrees = sample.graph.adjacency_csr.getnnz(axis=1)
    assert max(degrees) > np.median(degrees)
    assert len(set(sample.graph.hyperedges_truth)) == 40


def test_triangle_worlds_are_observationally_indistinguishable():
    first, second = triangle_ambiguity_example()
    assert first.hyperedges_truth != second.hyperedges_truth
    assert (first.adjacency_csr != second.adjacency_csr).nnz == 0
    assert (first.observed().adjacency_csr != second.observed().adjacency_csr).nnz == 0
    assert first.hyperedges_truth == [(0, 1, 2)]
    assert second.hyperedges_truth == [(0, 1), (0, 2), (1, 2)]


def test_noise_is_an_explicit_separate_protocol():
    sample = generate_synthetic(SyntheticConfig(case="S1", n_nodes=40,
                                                 num_hyperedges=24,
                                                 edge_drop_prob=0.15,
                                                 edge_add_prob=0.01), seed=21)
    assert sample.graph.provenance["observation_noisy"] is True
    assert sample.graph.observation_mode == "binary"
