"""Leakage-aware windows for real hyperedge event reconstruction.

Each output graph projects only its own *complete* source events and contains
only nodes observed in that window.  ``original_node_ids`` maps each window's
local node IDs back to source IDs.  Its truth
remains on the returned :class:`GraphRecord`, so callers must pass
``record.observed()`` to feature extraction and reserve truth for supervision
or evaluation.  These functions split retained events, not raw file rows that
may have been filtered by the loader.
"""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from numbers import Integral, Real
from typing import Optional, Sequence

import numpy as np

from .graph_io import GraphRecord
from .projection import project_hypergraph


def _finite_time(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric timestamp")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite numeric timestamp")
    return result


def _source_events(record: GraphRecord, *, require_times: bool) -> tuple[list[int], Optional[list[float]]]:
    if record.split is not None:
        raise ValueError("source graph must be unsplit")
    events = record.hyperedges_truth
    if not events:
        raise ValueError("source graph must contain retained hyperedge events")
    event_ids = record.hyperedge_event_ids
    if event_ids is None or len(event_ids) != len(events):
        raise ValueError("each source event requires an aligned event ID")
    if any(not isinstance(event_id, str) or not event_id for event_id in event_ids):
        raise ValueError("event IDs must be nonempty strings")
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("source event IDs must be unique")

    raw_times = record.hyperedge_timestamps
    if raw_times is None:
        if require_times:
            raise ValueError("timestamp split requires one timestamp per source event")
        return list(range(len(events))), None
    if len(raw_times) != len(events):
        raise ValueError("timestamps must align with source events")
    times = [_finite_time(value, f"event timestamp {index}")
             for index, value in enumerate(raw_times)]
    # Stable tie order preserves source order while making the chronology
    # independent of the order in which the file happened to list events.
    order = sorted(range(len(events)), key=lambda index: (times[index], index))
    return order, times


def _build_windows(record: GraphRecord, order_by_split: dict[str, list[int]],
                   times: Optional[list[float]], method: str,
                   bounds: dict[str, tuple[Optional[float | int], Optional[float | int]]],
                   *, split_roles: Optional[dict[str, Optional[str]]] = None,
                   window_indices: Optional[dict[str, int]] = None
                   ) -> dict[str, GraphRecord]:
    names = tuple(order_by_split)
    if not names or any(not order_by_split[name] for name in names):
        raise ValueError("requested windows must all contain events")
    all_positions = [index for name in names for index in order_by_split[name]]
    event_count = len(record.hyperedges_truth or [])
    if len(all_positions) != event_count or set(all_positions) != set(range(event_count)):
        raise ValueError("each source event must belong to exactly one window")

    source = {"graph_id": record.graph_id,
              "source_id": record.provenance.get("source_id", record.graph_id)}
    for key in ("source_url", "license", "files"):
        if key in record.provenance:
            source[key] = deepcopy(record.provenance[key])

    output: dict[str, GraphRecord] = {}
    for name in names:
        positions = order_by_split[name]
        source_nodes = sorted({node for index in positions
                               for node in record.hyperedges_truth[index]})
        source_to_local = {node: local for local, node in enumerate(source_nodes)}
        edges = [tuple(source_to_local[node] for node in record.hyperedges_truth[index])
                 for index in positions]
        event_ids = [record.hyperedge_event_ids[index] for index in positions]
        window_times = None if times is None else [times[index] for index in positions]
        lower, upper = bounds[name]
        window = {
            "method": method,
            "lower_inclusive": lower,
            "upper_exclusive": upper,
            "retained_event_count": len(positions),
            "first_timestamp": None if window_times is None else window_times[0],
            "last_timestamp": None if window_times is None else window_times[-1],
        }
        if window_indices is not None:
            window["index"] = window_indices[name]
        output[name] = GraphRecord(
            graph_id=f"{record.graph_id}::{name}",
            num_nodes=len(source_nodes),
            adjacency_csr=project_hypergraph(edges, len(source_nodes),
                                              mode=record.observation_mode),
            # A feature matrix built from the full source period could contain
            # future information; attach window-safe features separately.
            features=None,
            hyperedges_truth=edges,
            hyperedge_event_ids=event_ids,
            hyperedge_timestamps=window_times,
            observation_mode=record.observation_mode,
            split=name if split_roles is None else split_roles[name],
            provenance={"event_source": deepcopy(source), "window": window,
                        "node_universe": "members_of_window_events",
                        "source_features_copied": False},
            original_node_ids=tuple(record.original_node_ids[node]
                                    for node in source_nodes),
        )
    return output


def split_events_into_windows(record: GraphRecord, *, boundaries: Sequence[Real],
                              labels: Optional[Sequence[Optional[str]]] = None
                              ) -> list[GraphRecord]:
    """Partition complete events into any number of nonempty time windows.

    ``boundaries`` are prespecified, strictly increasing numeric timestamps.
    For boundaries ``[b0, b1]``, the windows are ``t < b0``,
    ``b0 <= t < b1``, and ``t >= b1``.  Equal-timestamp events therefore
    always belong to the same window.  ``labels`` optionally assigns each
    window a ``train``, ``validation``, or ``test`` role; roles may repeat
    across windows.  With no labels, ``GraphRecord.split`` is ``None``.
    Outputs are ordered chronologically, with local node IDs and source IDs
    recoverable through each record's ``original_node_ids``.
    """
    if isinstance(boundaries, (str, bytes)):
        raise ValueError("boundaries must be a sequence of numeric timestamps")
    cuts = [_finite_time(value, f"boundary {index}")
            for index, value in enumerate(boundaries)]
    if any(left >= right for left, right in zip(cuts, cuts[1:])):
        raise ValueError("boundaries must be strictly increasing")
    window_count = len(cuts) + 1
    if labels is None:
        roles: list[Optional[str]] = [None] * window_count
    else:
        if isinstance(labels, (str, bytes)) or len(labels) != window_count:
            raise ValueError("labels must have one role per window")
        roles = list(labels)
        if any(role not in {None, "train", "validation", "test"} for role in roles):
            raise ValueError("window labels must be train, validation, test, or None")

    order, times = _source_events(record, require_times=True)
    assert times is not None
    names = [f"window:{index}" for index in range(window_count)]
    partitions = {name: [] for name in names}
    for event_index in order:
        partitions[names[bisect_right(cuts, times[event_index])]].append(event_index)
    bounds = {name: (None if index == 0 else cuts[index - 1],
                     None if index == len(cuts) else cuts[index])
              for index, name in enumerate(names)}
    split_roles = dict(zip(names, roles))
    indices = {name: index for index, name in enumerate(names)}
    records = _build_windows(record, partitions, times, "timestamp_boundaries", bounds,
                             split_roles=split_roles, window_indices=indices)
    return [records[name] for name in names]


def split_events_by_time(record: GraphRecord, *, train_end: Real,
                         validation_end: Real) -> dict[str, GraphRecord]:
    """Split by prespecified numeric time boundaries.

    Train contains ``t < train_end``, validation contains
    ``train_end <= t < validation_end``, and test contains
    ``t >= validation_end``.  All equal-timestamp events stay together.  The
    bounds are recorded as half-open windows; ``None`` means an open end.
    """
    first = _finite_time(train_end, "train_end")
    second = _finite_time(validation_end, "validation_end")
    if first >= second:
        raise ValueError("train_end must precede validation_end")
    order, times = _source_events(record, require_times=True)
    assert times is not None
    partitions = {
        "train": [index for index in order if times[index] < first],
        "validation": [index for index in order if first <= times[index] < second],
        "test": [index for index in order if times[index] >= second],
    }
    bounds = {"train": (None, first),
              "validation": (first, second),
              "test": (second, None)}
    return _build_windows(record, partitions, times, "timestamp_boundaries", bounds)


def split_events_by_count(record: GraphRecord, *, train_count: int,
                          validation_count: int) -> dict[str, GraphRecord]:
    """Split retained events by prespecified counts in chronological order.

    With timestamps, an event-count boundary may not divide events sharing a
    timestamp.  Without timestamps, the source event order is used and the
    output provenance names this as an event-order split, not a time split.
    """
    for name, value in (("train_count", train_count),
                        ("validation_count", validation_count)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    order, times = _source_events(record, require_times=False)
    first = int(train_count)
    second = first + int(validation_count)
    if second >= len(order):
        raise ValueError("train, validation, and test windows must all contain events")
    if times is not None:
        for boundary in (first, second):
            if times[order[boundary - 1]] == times[order[boundary]]:
                raise ValueError("event-count boundary divides an equal-timestamp group")
    partitions = {"train": order[:first],
                  "validation": order[first:second],
                  "test": order[second:]}
    bounds = {"train": (0, first),
              "validation": (first, second),
              "test": (second, len(order))}
    method = "chronological_event_counts" if times is not None else "source_order_event_counts"
    return _build_windows(record, partitions, times, method, bounds)
