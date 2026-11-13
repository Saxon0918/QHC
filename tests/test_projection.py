import numpy as np
import pytest

from qhyper.graph_io import (CandidateRecord, GraphRecord, candidate_id,
                             canonicalize_edge, load_cornell_event_files)
from qhyper.projection import incidence_matrix, project_hypergraph


def test_binary_and_multiplicity_projection_preserve_event_semantics():
    events = [(2, 0, 1), (0, 1, 2), (1, 3)]
    A = project_hypergraph(events, 4, mode="binary").toarray()
    W = project_hypergraph(events, 4, mode="multiplicity").toarray()
    assert A[0, 1] == A[0, 2] == A[1, 2] == A[1, 3] == 1
    assert W[0, 1] == W[0, 2] == W[1, 2] == 2
    assert W[1, 3] == 1
    assert np.array_equal(A, A.T)
    assert np.array_equal(W, W.T)
    assert np.all(np.diag(A) == 0)
    assert np.all(np.diag(W) == 0)
    B = incidence_matrix(events, 4)
    assert B.shape == (4, 3)
    assert incidence_matrix(events, 4, deduplicate=True).shape == (4, 2)


def test_graph_record_hides_truth_from_observed_input():
    events = [(0, 1, 2)]
    A = project_hypergraph(events, 3)
    record = GraphRecord("a", 3, A, hyperedges_truth=events,
                         hyperedge_event_ids=["evt-1"],
                         original_node_ids=("u", "v", "w"))
    observed = record.observed()
    assert observed.hyperedges_truth is None
    assert observed.hyperedge_event_ids is None
    assert observed.hyperedge_timestamps is None
    assert observed.original_to_internal() == {"u": 0, "v": 1, "w": 2}
    assert (observed.adjacency_csr != A).nnz == 0
    assert record.hyperedges_truth == [(0, 1, 2)]


def test_canonical_ids_and_safe_candidate_feature_payload():
    assert canonicalize_edge([2, 0, 1]) == (0, 1, 2)
    assert candidate_id("g", [2, 0, 1]) == candidate_id("g", [0, 1, 2])
    assert candidate_id("g", [0, 1, 2]) != candidate_id("other", [0, 1, 2])
    record = CandidateRecord("g", (2, 0, 1), ("local_clique",),
                             classical_features=np.array([1.0]),
                             label_if_training=1, prediction_score=0.9)
    payload = record.feature_payload()
    assert payload["node_ids"] == [0, 1, 2]
    assert "label_if_training" not in payload
    assert "prediction_score" not in payload


def test_invalid_hyperedges_and_projection_modes_rejected():
    with pytest.raises(ValueError):
        canonicalize_edge([0, 0, 1])
    with pytest.raises(ValueError):
        project_hypergraph([(0, 3)], 3)
    with pytest.raises(ValueError):
        project_hypergraph([(0, 1)], 3, mode="unknown")
    with pytest.raises(ValueError):
        GraphRecord("bad", 3, project_hypergraph([(0, 1)], 3) * 2)


def test_cornell_event_loader_requires_exact_versioned_provenance(tmp_path):
    nverts = tmp_path / "same-nverts.txt"
    simplices = tmp_path / "same-simplices.txt"
    times = tmp_path / "same-times.txt"
    nverts.write_text("3\n2\n", encoding="utf-8")
    simplices.write_text("10\n20\n30\n20\n40\n", encoding="utf-8")
    times.write_text("100\n101\n", encoding="utf-8")
    temporal = load_cornell_event_files(
        str(nverts), str(simplices), "cornell/email-Enron/temporal",
        "https://example.test/cornell", str(times))
    other = load_cornell_event_files(
        str(nverts), str(simplices), "other/email-Enron/graph",
        "https://example.test/other", str(times))
    assert temporal.graph_id != other.graph_id
    assert temporal.hyperedges_truth == [(0, 1, 2), (1, 3)]
    assert temporal.original_node_ids == (10, 20, 30, 40)
    assert temporal.hyperedge_timestamps == [100.0, 101.0]
    assert temporal.provenance["files"]["simplices"]["sha256"]
    with pytest.raises(ValueError):
        load_cornell_event_files(str(nverts), str(simplices), "cornell/email-Enron",
                                 "https://example.test")


def test_cornell_loader_filters_singletons_with_original_event_ids(tmp_path):
    nverts = tmp_path / "nverts.txt"
    simplices = tmp_path / "simplices.txt"
    times = tmp_path / "times.txt"
    nverts.write_text("1\n2\n", encoding="utf-8")
    simplices.write_text("9\n10\n11\n", encoding="utf-8")
    times.write_text("5\n6\n", encoding="utf-8")
    record = load_cornell_event_files(
        str(nverts), str(simplices), "cornell/test/temporal",
        "https://example.test/test", str(times))
    assert record.num_nodes == 3
    assert record.hyperedges_truth == [(1, 2)]
    assert record.hyperedge_event_ids == ["cornell/test/temporal:event:1"]
    assert record.hyperedge_timestamps == [6.0]
    assert record.provenance["excluded_small_event_count"] == 1
