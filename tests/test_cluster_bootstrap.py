"""The independent graph/time block, not the candidate, is the resampling unit."""

from itertools import product

import numpy as np
from sklearn.metrics import average_precision_score

from qhyper.metrics import paired_bootstrap_delta


def test_four_graph_interval_enumerates_all_cluster_draws():
    labels = np.array([0, 1, 0, 1] * 4)
    classical = np.array([0.2, 0.6, 0.4, 0.5,
                          0.1, 0.8, 0.3, 0.7,
                          0.4, 0.5, 0.2, 0.6,
                          0.3, 0.6, 0.4, 0.7])
    quantum = np.array([0.1, 0.8, 0.2, 0.9,
                        0.4, 0.7, 0.3, 0.6,
                        0.5, 0.4, 0.2, 0.6,
                        0.3, 0.5, 0.4, 0.7])
    groups = np.repeat(["a", "b", "c", "d"], 4)
    result = paired_bootstrap_delta(labels, classical, quantum, groups,
                                    n_bootstrap=3, seed=1)
    reference = []
    for draw in product(range(4), repeat=4):
        indices = np.concatenate([np.arange(4 * i, 4 * i + 4) for i in draw])
        reference.append(average_precision_score(labels[indices], quantum[indices])
                         - average_precision_score(labels[indices], classical[indices]))
    expected = np.quantile(reference, [0.025, 0.975])
    assert result["bootstrap_method"] == "exact_enumeration"
    assert result["bootstrap_total_combinations"] == 256
    assert result["bootstrap_attempted"] == result["bootstrap_valid"] == 256
    np.testing.assert_allclose([result["ci_low"], result["ci_high"]], expected)
    assert result["tail_probability_mc_se"] is None
    assert "exploratory" in result["inference_warning"]
    # The exact interval has no Monte Carlo seed or requested-count dependence.
    other = paired_bootstrap_delta(labels, classical, quantum, groups,
                                   n_bootstrap=20, seed=999)
    assert [result["ci_low"], result["ci_high"]] == [other["ci_low"], other["ci_high"]]


def test_six_graphs_use_reported_monte_carlo_precision():
    labels = np.tile([0, 1], 6)
    classical = np.tile([0.4, 0.6], 6)
    quantum = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7,
                        0.5, 0.5, 0.7, 0.3, 0.9, 0.1])
    groups = np.repeat(np.arange(6), 2)
    result = paired_bootstrap_delta(labels, classical, quantum, groups,
                                    n_bootstrap=50, seed=5, metric="brier")
    assert result["bootstrap_method"] == "monte_carlo"
    assert result["bootstrap_total_combinations"] is None
    assert result["bootstrap_attempted"] == result["bootstrap_valid"] == 50
    assert result["tail_probability_mc_se"] > 0
    rng = np.random.default_rng(5)
    group_delta = np.array([
        np.mean((labels[2*i:2*i+2] - classical[2*i:2*i+2]) ** 2)
        - np.mean((labels[2*i:2*i+2] - quantum[2*i:2*i+2]) ** 2)
        for i in range(6)
    ])
    draws = [np.mean(group_delta[rng.integers(0, 6, size=6)]) for _ in range(50)]
    assert result["ci_low"] < result["ci_high"]
    np.testing.assert_allclose([result["ci_low"], result["ci_high"]],
                               np.quantile(draws, [0.025, 0.975]))


def test_one_group_does_not_pretend_to_estimate_uncertainty():
    result = paired_bootstrap_delta([0, 1], [0.4, 0.6], [0.3, 0.7], ["one", "one"])
    assert result["delta"] is not None
    assert result["ci_low"] is None and result["ci_high"] is None
    assert result["group_count"] == 1
    assert "cannot estimate" in result["inference_warning"]
