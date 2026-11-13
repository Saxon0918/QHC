"""Complete-event time windows must not leak source events across graphs."""

from __future__ import annotations

import numpy as np
import pytest

from qhyper.graph_io import GraphRecord
from qhyper.projection import project_hypergraph
from qhyper.real_data import (split_events_by_count, split_events_by_time,
                              split_events_into_windows)


def _record(events, *, times=None, event_ids=None, mode="binary") -> GraphRecord:
    return GraphRecord(
        graph_id="cornell/toy/temporal-v1",
        num_nodes=5,
        adjacency_csr=project_hypergraph(events, 5, mode=mode),
        features=np.ones((5, 2)),
        hyperedges_truth=list(events),
        hyperedge_event_ids=event_ids or [f"raw:{i}" for i in range(len(events))],
        hyperedge_timestamps=times,
        observation_mode=mode,
        original_node_ids=(10, 20, 30, 40, 99),
        provenance={"source_id": "cornell/toy/temporal-v1",
                    "source_url": "https://example.test/cornell/toy",
                    "files": {"times": {"sha256": "abc"}}},
    )


def test_time_windows_keep_complete_events_ties_and_local_projections():
    # Deliberately unsorted input, with two repeated member sets at t=20/30.
    events = [(0, 1), (0, 2), (0, 1), (2, 3), (1, 3), (0, 3)]
    record = _record(events, times=[30, 10, 20, 20, 30, 40])
    windows = split_events_by_time(record, train_end=20, validation_end=40)
    assert [windows[name].hyperedge_event_ids for name in
            ("train", "validation", "test")] == [
                ["raw:1"], ["raw:2", "raw:3", "raw:0", "raw:4"], ["raw:5"]]
    assert windows["validation"].hyperedges_truth.count((0, 1)) == 2
    assert windows["train"].original_node_ids == (10, 30)
    assert windows["train"].num_nodes == 2
    assert 20 not in windows["train"].original_node_ids
    assert 40 not in windows["train"].original_node_ids
    assert windows["validation"].adjacency_csr[0, 2] == 0
    assert windows["test"].original_node_ids == (10, 40)
    for name, window in windows.items():
        assert window.split == name
        assert set(window.original_node_ids) <= set(record.original_node_ids)
        assert window.num_nodes == len(window.original_node_ids)
        assert window.features is None
        assert (window.adjacency_csr != project_hypergraph(
            window.hyperedges_truth, window.num_nodes)).nnz == 0
        for edge, event_id in zip(window.hyperedges_truth, window.hyperedge_event_ids):
            source_edge = record.hyperedges_truth[int(event_id.split(":")[1])]
            assert {window.original_node_ids[node] for node in edge} == {
                record.original_node_ids[node] for node in source_edge}
        assert window.observed().hyperedges_truth is None
        assert window.observed().hyperedge_event_ids is None
        assert window.provenance["event_source"]["files"]["times"]["sha256"] == "abc"
    assert windows["validation"].provenance["window"] == {
        "method": "timestamp_boundaries", "lower_inclusive": 20.0,
        "upper_exclusive": 40.0, "retained_event_count": 4,
        "first_timestamp": 20.0, "last_timestamp": 30.0}
    all_ids = [event_id for window in windows.values()
               for event_id in window.hyperedge_event_ids]
    assert len(all_ids) == len(set(all_ids)) == len(events)


def test_count_windows_without_timestamps_use_source_order():
    events = [(0, 1), (0, 1), (1, 2), (2, 3), (3, 4)]
    windows = split_events_by_count(_record(events), train_count=2,
                                    validation_count=1)
    assert [windows[name].hyperedge_event_ids for name in
            ("train", "validation", "test")] == [
                ["raw:0", "raw:1"], ["raw:2"], ["raw:3", "raw:4"]]
    assert windows["train"].provenance["window"]["method"] == "source_order_event_counts"
    assert windows["test"].provenance["window"]["upper_exclusive"] == 5
    assert windows["train"].hyperedge_timestamps is None


def test_count_windows_reject_boundary_inside_timestamp_tie():
    events = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
    record = _record(events, times=[1, 2, 2, 3, 4])
    with pytest.raises(ValueError, match="equal-timestamp"):
        split_events_by_count(record, train_count=2, validation_count=1)
    windows = split_events_by_count(record, train_count=1, validation_count=2)
    assert windows["validation"].hyperedge_timestamps == [2.0, 2.0]
    assert windows["test"].hyperedge_event_ids == ["raw:3", "raw:4"]


def test_multiplicity_window_counts_repeated_events():
    record = _record([(0, 1), (0, 1), (1, 2), (2, 3)],
                     times=[1, 1, 2, 3], mode="multiplicity")
    windows = split_events_by_time(record, train_end=2, validation_end=3)
    assert windows["train"].adjacency_csr[0, 1] == 2
    assert windows["validation"].original_node_ids == (20, 30)
    assert windows["validation"].adjacency_csr[0, 1] == 1


def test_arbitrary_time_windows_keep_ties_complete_events_and_disjoint_ids():
    events = [(0, 1), (2, 3), (0, 2), (1, 3), (3, 4),
              (1, 4), (0, 1), (2, 4)]
    record = _record(events, times=[5, 1, 2, 5, 3, 6, 2, 4])
    windows = split_events_into_windows(
        record, boundaries=[2, 3, 5, 6],
        labels=["train", "train", "validation", "test", "test"])
    assert len(windows) == 5
    assert [window.split for window in windows] == [
        "train", "train", "validation", "test", "test"]
    assert [window.hyperedge_event_ids for window in windows] == [
        ["raw:1"], ["raw:2", "raw:6"], ["raw:4", "raw:7"],
        ["raw:0", "raw:3"], ["raw:5"]]
    assert [window.provenance["window"]["index"] for window in windows] == list(range(5))
    assert windows[1].provenance["window"]["lower_inclusive"] == 2.0
    assert windows[1].provenance["window"]["upper_exclusive"] == 3.0
    assert 10 not in windows[0].original_node_ids
    assert 99 not in windows[0].original_node_ids
    all_ids = []
    for window in windows:
        assert window.num_nodes == len(window.original_node_ids)
        assert (window.adjacency_csr != project_hypergraph(
            window.hyperedges_truth, window.num_nodes)).nnz == 0
        for edge, event_id in zip(window.hyperedges_truth, window.hyperedge_event_ids):
            source_edge = record.hyperedges_truth[int(event_id.split(":")[1])]
            assert {window.original_node_ids[node] for node in edge} == {
                record.original_node_ids[node] for node in source_edge}
            all_ids.append(event_id)
    assert len(all_ids) == len(set(all_ids)) == len(events)


def test_arbitrary_windows_reject_empty_or_out_of_order_cuts():
    record = _record([(0, 1), (1, 2), (2, 3)], times=[1, 2, 3])
    assert [window.split for window in split_events_into_windows(
        record, boundaries=[2])] == [None, None]
    with pytest.raises(ValueError, match="requested windows must all contain events"):
        split_events_into_windows(record, boundaries=[1.5, 1.75])
    with pytest.raises(ValueError, match="strictly increasing"):
        split_events_into_windows(record, boundaries=[2, 2])
    with pytest.raises(ValueError, match="one role per window"):
        split_events_into_windows(record, boundaries=[2], labels=["train"])


@pytest.mark.parametrize("times", [None, [1, 2, None], [1, 2, np.nan]])
def test_time_split_requires_complete_finite_timestamps(times):
    record = _record([(0, 1), (1, 2), (2, 3)], times=times)
    with pytest.raises(ValueError, match="timestamp"):
        split_events_by_time(record, train_end=2, validation_end=3)


def test_duplicate_event_ids_and_empty_windows_are_rejected():
    record = _record([(0, 1), (1, 2), (2, 3)], times=[1, 2, 3],
                     event_ids=["same", "same", "last"])
    with pytest.raises(ValueError, match="event IDs must be unique"):
        split_events_by_time(record, train_end=2, validation_end=3)
    record.hyperedge_event_ids = ["one", "two", "three"]
    with pytest.raises(ValueError, match="must all contain events"):
        split_events_by_time(record, train_end=1, validation_end=3)
    with pytest.raises(ValueError, match="must all contain events"):
        split_events_by_count(record, train_count=1, validation_count=2)
