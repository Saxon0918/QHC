"""Conditional temporal-identification identities and deliberate failure cases."""

from __future__ import annotations

from itertools import combinations
from math import exp, isclose

import numpy as np
import pytest

from qhyper.projection import project_hypergraph
from qhyper.revision_temporal import (
    estimate_joint_absence_rate,
    gamma_poisson_predictive_presence,
    joint_absence_probability,
    joint_absence_rate,
    mobius_invert_superset_intensities,
    simulate_poisson_windows,
    superset_intensity_from_absence,
    theoretical_superset_intensity,
)


def test_one_binary_triangle_is_ambiguous_but_repeated_distribution_differs() -> None:
    one_triad = [(0, 1, 2)]
    three_pairs = [(0, 1), (0, 2), (1, 2)]
    assert (project_hypergraph(one_triad, 3, mode="binary") !=
            project_hypergraph(three_pairs, 3, mode="binary")).nnz == 0

    group_rates = {(0, 1, 2): 0.7}
    pair_rates = {(0, 1): 0.7, (0, 2): 0.7, (1, 2): 0.7}
    # Each dyad alone has the same absence probability under the two worlds.
    for pair in three_pairs:
        assert joint_absence_probability(group_rates, [pair], 2.0) == pytest.approx(
            joint_absence_probability(pair_rates, [pair], 2.0))
    # Their *joint* absence differs; independent pair events have no triad rate.
    assert joint_absence_probability(group_rates, [(0, 1), (0, 2)], 2.0) == pytest.approx(
        exp(-1.4))
    assert joint_absence_probability(pair_rates, [(0, 1), (0, 2)], 2.0) == pytest.approx(
        exp(-2.8))
    group_s = superset_intensity_from_absence(
        [(0, 1), (0, 2)], lambda pairs: joint_absence_rate(group_rates, pairs))
    pair_s = superset_intensity_from_absence(
        [(0, 1), (0, 2)], lambda pairs: joint_absence_rate(pair_rates, pairs))
    assert group_s == pytest.approx(0.7)
    assert pair_s == pytest.approx(0.0)


def test_inclusion_exclusion_and_mobius_recover_bounded_sparse_rates() -> None:
    rates = {
        (0, 1): 0.1,
        (0, 1, 2): 0.4,
        (0, 1, 2, 3): 0.2,
        (0, 1, 2, 3, 4): 0.3,
    }
    covers = {
        (0, 1, 2): [(0, 1), (1, 2)],
        (0, 1, 2, 3): [(0, 1), (2, 3)],
        (0, 1, 2, 3, 4): [(0, 1), (2, 3), (3, 4)],
    }
    for vertices, cover in covers.items():
        from_absence = superset_intensity_from_absence(
            cover, lambda pairs: joint_absence_rate(rates, pairs))
        assert from_absence == pytest.approx(theoretical_superset_intensity(rates, vertices))

    universe = [edge for size in range(2, 6)
                for edge in combinations(range(5), size)]
    superset = {edge: theoretical_superset_intensity(rates, edge) for edge in universe}
    recovered = mobius_invert_superset_intensities(superset, universe)
    assert recovered.keys() == superset.keys()
    for edge in universe:
        assert recovered[edge] == pytest.approx(rates.get(edge, 0.0), abs=1e-12)


def test_simulator_has_binary_projection_and_empirical_identity() -> None:
    rates = {(0, 1, 2): 0.5, (0, 1): 0.2}
    sample = simulate_poisson_windows(rates, n_nodes=3, duration=0.4,
                                      n_windows=5000, seed=9)
    assert sample.latent_counts.shape == (5000, 2)
    assert sample.pair_counts.shape == sample.binary_adjacency.shape == (5000, 3, 3)
    assert np.array_equal(sample.binary_adjacency, (sample.pair_counts > 0).astype(np.int8))
    empirical = estimate_joint_absence_rate(
        sample.binary_adjacency, [(0, 1), (0, 2)], duration=0.4)
    assert empirical == pytest.approx(joint_absence_rate(rates, [(0, 1), (0, 2)]),
                                      abs=0.09)


def test_empirical_estimator_refuses_current_eight_window_pilot_and_zero_absences() -> None:
    eight = np.zeros((8, 3, 3), dtype=np.int8)
    with pytest.raises(ValueError, match="insufficient repeated windows"):
        estimate_joint_absence_rate(eight, [(0, 1)], duration=1.0)
    all_present = np.ones((100, 3, 3), dtype=np.int8)
    all_present[:, np.arange(3), np.arange(3)] = 0
    with pytest.raises(ValueError, match="too few joint-absence"):
        estimate_joint_absence_rate(all_present, [(0, 1)], duration=1.0)


def test_gamma_poisson_prior_uses_only_training_counts_and_exposures() -> None:
    probability = gamma_poisson_predictive_presence(
        [0, 2, 1], [1.0, 2.0, 0.5], prior_shape=1.5, prior_rate=2.0,
        future_exposure=1.2)
    shape, rate = 4.5, 5.5
    assert isclose(probability, 1.0 - (rate / (rate + 1.2)) ** shape)
    assert 0 < probability < 1
    with pytest.raises(ValueError, match="training counts"):
        gamma_poisson_predictive_presence(
            [0, -1], [1.0, 1.0], prior_shape=1.0, prior_rate=1.0,
            future_exposure=1.0)


def test_mobius_rejects_incomplete_or_inconsistent_rates() -> None:
    with pytest.raises(ValueError, match="one superset intensity"):
        mobius_invert_superset_intensities({(0, 1): 0.1}, [(0, 1), (0, 1, 2)])
    with pytest.raises(ValueError, match="negative inverted rate"):
        mobius_invert_superset_intensities(
            {(0, 1, 2): 1.0, (0, 1): 0.2}, [(0, 1), (0, 1, 2)])
