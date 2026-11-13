"""Fixed-candidate scoring, leakage boundary, and exact recovery metrics."""

import numpy as np
import pytest

from qhyper.metrics import (
    candidate_recall,
    evaluate_candidate_predictions,
    evaluate_hyperedge_sets,
    matched_edge_jaccard,
    paired_bootstrap_delta,
)
from qhyper.scorer import (
    choose_validation_threshold,
    fit_candidate_scorer,
    predict_candidate_scores,
    select_hyperedges,
)


def test_scorer_scaler_fits_training_rows_only():
    train_features = np.array([[0, 1], [1, 1], [3, 0], [4, 0]], dtype=float)
    train_labels = [0, 0, 1, 1]
    model = fit_candidate_scorer(train_features, train_labels)
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, train_features.mean(axis=0))
    probabilities = predict_candidate_scores(model, np.array([[1e9, 1], [-1e9, 0]]))
    assert probabilities.shape == (2,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, train_features.mean(axis=0))
    with pytest.raises(ValueError):
        fit_candidate_scorer(train_features, [1, 1, 1, 1])


def test_validation_threshold_and_selection():
    validation = choose_validation_threshold([0, 1, 0, 1], [0.2, 0.9, 0.3, 0.8])
    assert validation["threshold"] == pytest.approx(0.8)
    assert validation["validation_f1"] == 1.0
    edges = select_hyperedges([(2, 1), (0, 1), (1, 2)], [0.9, 0.2, 0.8], 0.8)
    assert edges == [(1, 2)]
    with pytest.raises(ValueError):
        select_hyperedges([(1, 1, 2)], [0.8], 0.5)
    empty = choose_validation_threshold([0, 0], [0.3, 0.4])
    assert empty["selected_count"] == 0


def test_candidate_and_exact_recovery_metrics():
    candidates = [(0, 1), (0, 1, 2), (0, 2, 3)]
    truth = [(0, 1), (0, 1, 2), (1, 2, 3)]
    coverage = candidate_recall(candidates, truth)
    assert coverage["recall"] == pytest.approx(2 / 3)
    assert coverage["by_cardinality"]["3"]["recall"] == pytest.approx(0.5)
    predicted = [(0, 1), (0, 2, 3)]
    metrics = evaluate_hyperedge_sets(predicted, truth)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 2
    assert metrics["recall"] <= coverage["recall"]
    assert metrics["f1"] == pytest.approx(0.4)
    assert metrics["jaccard"] == pytest.approx(0.25)
    assert evaluate_hyperedge_sets([], [])["f1"] == 1.0
    assert evaluate_hyperedge_sets([], [(0, 1)])["f1"] == 0.0
    assert evaluate_hyperedge_sets([(0, 1)], [])["recall"] == 1.0


def test_discrimination_and_group_bootstrap():
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    base = [0.4, 0.6, 0.5, 0.55, 0.4, 0.6, 0.5, 0.55]
    proposed = [0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8]
    summary = evaluate_candidate_predictions(labels, proposed, sizes=[2, 2, 3, 3] * 2, budget=4)
    assert summary["ap"] == 1.0
    assert summary["positive_rate"] == 0.5
    assert summary["precision_at_budget"] == 1.0
    assert set(summary["by_cardinality"]) == {"2", "3"}
    interval = paired_bootstrap_delta(labels, base, proposed, groups=["g1"] * 4 + ["g2"] * 4, n_bootstrap=30, seed=1)
    assert interval["delta"] >= 0
    assert interval["bootstrap_method"] == "exact_enumeration"
    assert interval["bootstrap_total_combinations"] == 4
    assert interval["bootstrap_valid"] == 4
    assert interval["ci_low"] is not None
    with pytest.raises(ValueError):
        evaluate_candidate_predictions([0, 1], [0.2])


def test_approximate_match_is_separate_from_exact():
    approximate = matched_edge_jaccard([(0, 1, 2)], [(0, 1, 3)], minimum_jaccard=0.5)
    exact = evaluate_hyperedge_sets([(0, 1, 2)], [(0, 1, 3)])
    assert approximate["matched_count"] == 1
    assert approximate["mean_matched_jaccard"] == 0.5
    assert exact["tp"] == 0
