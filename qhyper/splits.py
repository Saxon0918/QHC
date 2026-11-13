"""Split checks that make the intended cross-graph protocol explicit."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, Sequence, Set


def validate_disjoint_graph_splits(splits: Mapping[str, Sequence[str]]) -> None:
    """Raise if an observed graph is assigned to more than one split."""
    owner: Dict[str, str] = {}
    for split, graph_ids in splits.items():
        for graph_id in graph_ids:
            if graph_id in owner:
                raise ValueError("graph {!r} appears in both {!r} and {!r}".format(
                    graph_id, owner[graph_id], split))
            owner[graph_id] = split


def validate_event_disjointness(
    event_ids_by_split: Mapping[str, Iterable[str]],
) -> None:
    """Raise if one raw event contributes to different data splits."""
    owners: Dict[str, Set[str]] = defaultdict(set)
    for split, event_ids in event_ids_by_split.items():
        for event_id in event_ids:
            owners[str(event_id)].add(split)
    overlap = {event: sorted(parts) for event, parts in owners.items()
               if len(parts) > 1}
    if overlap:
        examples = list(sorted(overlap.items()))[:5]
        raise ValueError("raw events cross split boundaries: {}".format(examples))
