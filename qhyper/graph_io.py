"""Validated graph and candidate records for reconstruction experiments.

Ground-truth hyperedges are stored separately from the observable graph fields.
Call :meth:`GraphRecord.observed` before candidate or feature construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse


Edge = Tuple[int, ...]


def canonicalize_edge(nodes: Iterable[int], n_nodes: Optional[int] = None,
                      min_size: int = 2) -> Edge:
    """Return a sorted node tuple, rejecting duplicate and invalid memberships."""
    values = tuple(nodes)
    if any(isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer))
           for node in values):
        raise TypeError("hyperedge node IDs must be integers")
    edge = tuple(sorted(int(node) for node in values))
    if len(edge) < min_size:
        raise ValueError("hyperedge is smaller than the required minimum size")
    if len(set(edge)) != len(edge):
        raise ValueError("a hyperedge cannot contain a node more than once")
    if edge and (edge[0] < 0 or (n_nodes is not None and edge[-1] >= n_nodes)):
        raise ValueError("hyperedge node ID is outside [0, n_nodes)")
    return edge


def canonicalize_edges(hyperedges: Iterable[Iterable[int]],
                       n_nodes: Optional[int] = None,
                       deduplicate: bool = False,
                       min_size: int = 2) -> list[Edge]:
    """Normalize hyperedges; optionally collapse repeated *events* by membership."""
    normalized = [canonicalize_edge(edge, n_nodes=n_nodes, min_size=min_size)
                  for edge in hyperedges]
    return sorted(set(normalized)) if deduplicate else normalized


def candidate_id(graph_id: str, nodes: Iterable[int]) -> str:
    """Stable identifier scoped to one graph, independent of Python hash seed."""
    edge = canonicalize_edge(nodes)
    payload = json.dumps([graph_id, edge], separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class GraphRecord:
    """One graph observation and optional evaluation-only hyperedge truth.

    ``hyperedges_truth`` is an event list and may repeat a member set. Exact
    reconstruction metrics must explicitly deduplicate it. ``original_node_ids``
    preserves the inverse map from internal 0-based IDs to source IDs.
    """

    graph_id: str
    num_nodes: int
    adjacency_csr: sparse.csr_matrix
    features: Optional[np.ndarray] = None
    hyperedges_truth: Optional[list[Edge]] = None
    hyperedge_event_ids: Optional[list[str]] = None
    hyperedge_timestamps: Optional[list[float]] = None
    observation_mode: str = "binary"
    split: Optional[str] = None
    provenance: dict[str, Any] = field(default_factory=dict)
    original_node_ids: Optional[tuple[Any, ...]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.graph_id, str) or not self.graph_id:
            raise ValueError("graph_id must be a nonempty string")
        if not isinstance(self.num_nodes, int) or self.num_nodes < 0:
            raise ValueError("num_nodes must be a nonnegative integer")
        if self.observation_mode not in {"binary", "multiplicity"}:
            raise ValueError("observation_mode must be binary or multiplicity")
        A = sparse.csr_matrix(self.adjacency_csr, dtype=float)
        if A.shape != (self.num_nodes, self.num_nodes):
            raise ValueError("adjacency shape does not match num_nodes")
        A.sum_duplicates()
        A.eliminate_zeros()
        if A.nnz and (not np.isfinite(A.data).all() or np.any(A.data < 0)):
            raise ValueError("adjacency must be finite and nonnegative")
        if A.diagonal().any():
            raise ValueError("adjacency must have zero diagonal")
        if (A - A.T).nnz and not np.allclose((A - A.T).data, 0, atol=1e-12):
            raise ValueError("adjacency must be symmetric")
        if self.observation_mode == "binary" and A.nnz and not np.all(A.data == 1):
            raise ValueError("binary adjacency must have only 0/1 entries")
        self.adjacency_csr = A

        if self.features is not None:
            X = np.asarray(self.features, dtype=float)
            if X.ndim != 2 or X.shape[0] != self.num_nodes or not np.isfinite(X).all():
                raise ValueError("features must be a finite (N,d) array")
            self.features = X
        if self.hyperedges_truth is not None:
            self.hyperedges_truth = canonicalize_edges(self.hyperedges_truth,
                                                       n_nodes=self.num_nodes)
            event_count = len(self.hyperedges_truth)
            if self.hyperedge_event_ids is not None and len(self.hyperedge_event_ids) != event_count:
                raise ValueError("event IDs must align with hyperedges_truth")
            if self.hyperedge_timestamps is not None and len(self.hyperedge_timestamps) != event_count:
                raise ValueError("timestamps must align with hyperedges_truth")
        elif self.hyperedge_event_ids is not None or self.hyperedge_timestamps is not None:
            raise ValueError("event metadata requires hyperedges_truth")
        if self.original_node_ids is None:
            self.original_node_ids = tuple(range(self.num_nodes))
        elif len(self.original_node_ids) != self.num_nodes or len(set(self.original_node_ids)) != self.num_nodes:
            raise ValueError("original_node_ids must be a one-to-one map of length N")
        if self.split not in {None, "train", "validation", "test"}:
            raise ValueError("split must be train, validation, test, or None")

    def observed(self) -> GraphRecord:
        """Return only fields allowed as model input; evaluation truth is removed."""
        return GraphRecord(
            graph_id=self.graph_id,
            num_nodes=self.num_nodes,
            adjacency_csr=self.adjacency_csr.copy(),
            features=None if self.features is None else self.features.copy(),
            observation_mode=self.observation_mode,
            split=self.split,
            provenance=dict(self.provenance),
            original_node_ids=self.original_node_ids,
        )

    def original_to_internal(self) -> dict[Any, int]:
        """Return the source-ID to 0-based internal-ID map."""
        return {source_id: i for i, source_id in enumerate(self.original_node_ids)}


@dataclass
class CandidateRecord:
    """Candidate membership and provenance; model features and labels are optional."""

    graph_id: str
    node_ids: Edge
    proposal_sources: tuple[str, ...]
    classical_features: Optional[np.ndarray] = None
    quantum_features: Optional[np.ndarray] = None
    label_if_training: Optional[int] = None
    prediction_score: Optional[float] = None
    candidate_id: str = field(init=False)
    size: int = field(init=False)

    def __post_init__(self) -> None:
        self.node_ids = canonicalize_edge(self.node_ids)
        self.size = len(self.node_ids)
        self.candidate_id = candidate_id(self.graph_id, self.node_ids)
        self.proposal_sources = tuple(sorted(set(self.proposal_sources)))
        if not self.proposal_sources:
            raise ValueError("at least one proposal source is required")
        if self.label_if_training not in {None, 0, 1}:
            raise ValueError("label_if_training must be 0, 1 or None")
        for field_name in ("classical_features", "quantum_features"):
            value = getattr(self, field_name)
            if value is not None:
                arr = np.asarray(value, dtype=float)
                if arr.ndim != 1 or not np.isfinite(arr).all():
                    raise ValueError(f"{field_name} must be a finite 1-D array")
                setattr(self, field_name, arr)

    def feature_payload(self) -> dict[str, Any]:
        """Safe cache payload: intentionally excludes label and prediction score."""
        return {
            "graph_id": self.graph_id,
            "candidate_id": self.candidate_id,
            "node_ids": list(self.node_ids),
            "proposal_sources": list(self.proposal_sources),
            "classical_features": None if self.classical_features is None else self.classical_features.tolist(),
            "quantum_features": None if self.quantum_features is None else self.quantum_features.tolist(),
        }


def _integer_tokens(path: Path) -> list[int]:
    tokens = path.read_text(encoding="utf-8").split()
    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"expected integer tokens in {path}") from exc


def _numeric_tokens(path: Path) -> list[float]:
    tokens = path.read_text(encoding="utf-8").split()
    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"expected numeric tokens in {path}") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite numeric token in {path}")
    return values


def _file_digest(path: Path) -> dict[str, str]:
    return {"name": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def load_cornell_event_files(nverts_path: str, simplices_path: str,
                             source_id: str, source_url: str,
                             times_path: Optional[str] = None,
                             original_node_ids: Optional[Sequence[int]] = None,
                             license_name: Optional[str] = None,
                             min_event_size: int = 2) -> GraphRecord:
    """Read explicit Cornell ``nverts``/``simplices`` event files, without I/O to web.

    ``source_id`` must distinguish dataset and version, for example
    ``cornell/email-Enron/temporal``; a shared basename such as ``email-Enron``
    is never treated as sufficient provenance. ``original_node_ids`` can include
    isolated source nodes; otherwise the observed unique IDs define the node
    universe. Event files are read as *complete* events, never node-intersected.
    Singleton events are explicitly filtered at the declared minimum size and
    counted in provenance. This loader does not assign splits or use labels.
    """
    source_parts = source_id.split("/") if isinstance(source_id, str) else []
    if len(source_parts) < 3 or any(not part for part in source_parts) or not source_url:
        raise ValueError("source_id must be version-qualified and source_url required")
    if min_event_size < 2:
        raise ValueError("min_event_size must be at least two")
    nverts_file = Path(nverts_path)
    simplices_file = Path(simplices_path)
    nverts = _integer_tokens(nverts_file)
    flat = _integer_tokens(simplices_file)
    if not nverts or any(k < 1 for k in nverts):
        raise ValueError("nverts must list positive event sizes")
    if sum(nverts) != len(flat):
        raise ValueError("sum(nverts) must equal the number of simplices node IDs")
    timestamps = None
    files = {"nverts": _file_digest(nverts_file),
             "simplices": _file_digest(simplices_file)}
    if times_path is not None:
        times_file = Path(times_path)
        timestamps = _numeric_tokens(times_file)
        if len(timestamps) != len(nverts):
            raise ValueError("times must have one entry per event")
        files["times"] = _file_digest(times_file)
    if original_node_ids is None:
        original_ids = tuple(sorted(set(flat)))
    else:
        original_ids = tuple(original_node_ids)
        if len(set(original_ids)) != len(original_ids) or set(flat) - set(original_ids):
            raise ValueError("original_node_ids must uniquely include all event node IDs")
    source_to_internal = {source_id_value: i for i, source_id_value in enumerate(original_ids)}
    events: list[Edge] = []
    kept_ids: list[str] = []
    kept_times: Optional[list[float]] = [] if timestamps is not None else None
    offset = 0
    for event_index, k in enumerate(nverts):
        raw = flat[offset:offset + k]
        offset += k
        if k < min_event_size:
            continue
        events.append(canonicalize_edge((source_to_internal[node] for node in raw),
                                        n_nodes=len(original_ids)))
        kept_ids.append(f"{source_id}:event:{event_index}")
        if kept_times is not None:
            kept_times.append(timestamps[event_index])
    from .projection import project_hypergraph
    A = project_hypergraph(events, len(original_ids), mode="binary")
    return GraphRecord(
        graph_id=source_id,
        num_nodes=len(original_ids),
        adjacency_csr=A,
        hyperedges_truth=events,
        hyperedge_event_ids=kept_ids,
        hyperedge_timestamps=kept_times,
        observation_mode="binary",
        provenance={"source_id": source_id, "source_url": source_url,
                    "license": license_name, "files": files,
                    "event_semantics": "complete source event member set",
                    "min_event_size": min_event_size,
                    "raw_event_count": len(nverts),
                    "excluded_small_event_count": len(nverts) - len(events)},
        original_node_ids=original_ids,
    )
