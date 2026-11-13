import pytest

from qhyper.splits import validate_disjoint_graph_splits, validate_event_disjointness


def test_graph_splits_are_disjoint():
    validate_disjoint_graph_splits({"train": ["a", "b"], "test": ["c"]})
    with pytest.raises(ValueError):
        validate_disjoint_graph_splits({"train": ["a"], "test": ["a"]})


def test_raw_events_cannot_cross_splits():
    validate_event_disjointness({"train": ["e1"], "test": ["e2"]})
    with pytest.raises(ValueError):
        validate_event_disjointness({"train": ["e1"], "test": ["e1"]})
