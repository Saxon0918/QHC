import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from qhyper.hard_negatives import (
    match_hard_negatives_for_evaluation,
    matched_average_precision,
)


@pytest.mark.parametrize(
    ("degrees", "sources", "expected_level"),
    [
        ([1.0, 1.0], ["local", "local"], "size_degree_source"),
        ([1.0, 1.0], ["local", "maximal"], "size_degree"),
        ([1.0, 10.0], ["local", "local"], "size_source"),
        ([1.0, 10.0], ["local", "maximal"], "size_only"),
    ],
)
def test_matching_fallback_levels(degrees, sources, expected_level):
    result = match_hard_negatives_for_evaluation(
        [1, 0], [3, 3], degrees, sources, seed=42, n_degree_bins=2
    )
    assert result["positive_indices"] == [0]
    assert result["negative_indices"] == [1]
    assert result["fallback_counts"][expected_level] == 1
    assert sum(result["fallback_counts"].values()) == 1


def test_size_is_a_hard_constraint_and_unmatched_positives_are_reported():
    result = match_hard_negatives_for_evaluation(
        [1, 0], [3, 4], [1.0, 1.0], ["local", "local"]
    )
    assert result["matched_pair_count"] == 0
    assert result["unmatched_positive_indices"] == [0]
    assert result["by_size"]["3"]["unmatched_positive_count"] == 1
    assert matched_average_precision([1, 0], [0.9, 0.1], result)["matched_ap"] is None


def test_exact_match_is_reserved_before_fallback_positive():
    # Positive 0 has no exact source match; positive 1 does.  The exact pair
    # must be assigned first, even if input order puts the fallback first.
    result = match_hard_negatives_for_evaluation(
        labels=[1, 1, 0, 0],
        sizes=[3, 3, 3, 3],
        degree_means=[1.0, 1.0, 1.0, 10.0],
        proposal_sources=["other", "local", "local", "maximal"],
        seed=7,
        n_degree_bins=2,
    )
    assert result["matched_pair_count"] == 2
    exact = [record for record in result["pair_records"]
             if record["match_level"] == "size_degree_source"]
    assert len(exact) == 1
    assert exact[0]["positive_index"] == 1
    assert exact[0]["negative_index"] == 2


def test_reproducibility_no_reuse_and_matched_ap():
    labels = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.6, 0.7, 0.1, 0.3])
    sizes = np.array([3, 3, 3, 3, 3])
    degree = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    sources = [("local", "maximal")] * 5
    arguments = (labels, sizes, degree, sources)
    first = match_hard_negatives_for_evaluation(*arguments, seed=11)
    second = match_hard_negatives_for_evaluation(*arguments, seed=11)
    assert first == second
    assert len(set(first["negative_indices"])) == 2
    assert first["matched_pair_count"] == 2
    assert first["matched_positive_fraction"] == 1.0
    evaluated = matched_average_precision(labels, scores, first)
    selected = first["positive_indices"] + first["negative_indices"]
    assert evaluated["matched_ap"] == average_precision_score(labels[selected], scores[selected])
    assert evaluated["matched_positive_rate"] == 0.5
    assert evaluated["matched_candidate_count"] == 4


def test_partial_source_overlap_breaks_same_bin_fallback_tie():
    result = match_hard_negatives_for_evaluation(
        labels=[1, 0, 0],
        sizes=[3, 3, 3],
        degree_means=[1.0, 1.2, 1.1],
        proposal_sources=[("local", "maximal"), "local", "other"],
        n_degree_bins=1,
    )
    assert result["negative_indices"] == [1]
    assert result["pair_records"][0]["match_level"] == "size_degree"
    assert result["pair_records"][0]["proposal_source_jaccard"] == 0.5


def test_empty_or_single_class_candidate_pool_has_no_matched_ap():
    for labels, sizes, degrees, sources in [
        ([], [], [], []),
        ([1], [3], [1.0], ["local"]),
        ([0], [3], [1.0], ["local"]),
    ]:
        result = match_hard_negatives_for_evaluation(labels, sizes, degrees, sources)
        assert result["matched_pair_count"] == 0
        assert matched_average_precision(labels, [0.5] * len(labels), result)["matched_ap"] is None


def test_input_validation_and_label_guard():
    with pytest.raises(ValueError, match="aligned"):
        match_hard_negatives_for_evaluation([1, 0], [3], [1, 1], ["a", "a"])
    with pytest.raises(ValueError, match="nonnegative"):
        match_hard_negatives_for_evaluation([1, 0], [3, 3], [1, -1], ["a", "a"])
    with pytest.raises(ValueError, match="binary"):
        match_hard_negatives_for_evaluation([2, 0], [3, 3], [1, 1], ["a", "a"])
    matching = match_hard_negatives_for_evaluation([1, 0], [3, 3], [1, 1], ["a", "a"])
    with pytest.raises(ValueError, match="disagree"):
        matched_average_precision([0, 1], [0.9, 0.1], matching)
