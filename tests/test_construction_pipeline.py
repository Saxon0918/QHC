"""End-to-end B1 protocol checks across dense, sparse and archive inputs."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from qhyper.construction_pipeline import run_construction
from qhyper.construction_synthetic import make_synthetic_graph, stratified_node_split


def _config(*, dense_limit: int, hard: bool) -> dict:
    return {
        "project": {"protocol": "b1-test", "mode": "synthetic", "seed": 111},
        "synthetic": {"regimes": ["group"], "graph_seeds": [7], "n_nodes": 24},
        "kernel": {"kind": "norm_laplacian", "lam": 0.0,
                   "times": [0.7, 1.4], "weights": [1.0, 1.0],
                   "dense_limit": dense_limit},
        "sparse": {"landmarks": 12, "top_k": 4, "batch_size": 6},
        "hard": {"enabled": hard, "min_size": 2, "max_size": 4,
                 "neighborhood_cap": 6, "max_candidates": 30,
                 "budget": 3, "weight_grid": [0.5, 1.0]},
        "cluster": {"enabled": True, "n_groups": 6},
        "evaluation": {"C_grid": [0.1, 1.0]},
    }


def _run(tmp_path, config: dict):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "result"
    report = run_construction(path, output)
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["protocol"] == "b1-test"
    return report, output


def test_dense_b1_compares_matched_arms_and_certifies_error(tmp_path):
    report, output = _run(tmp_path, _config(dense_limit=32, hard=True))
    arms = {row["method"]: row for row in report["records"]}
    assert set(arms) == {
        "features_only", "graph_gcn", "quantum_direct", "markov_direct",
        "quantum_sparse", "markov_sparse", "quantum_hard", "markov_hard",
        "spread_markov_direct", "spread_markov_sparse", "spread_markov_hard",
        "quantum_cluster", "markov_cluster", "spread_markov_cluster",
    }
    assert report["summary"]["group"]["n_graph_records"] == 1
    assert report["summary"]["group"]["unit_type"] == "independent_synthetic_draw"
    assert set(report["summary"]["group"]["paired_quantum_minus_markov"]) == {
        "direct", "sparse", "cluster", "hard",
    }
    assert set(report["summary"]["group"]["paired_quantum_minus_spread_markov"]) == {
        "direct", "sparse", "cluster", "hard",
    }
    for family in ("quantum", "markov", "spread_markov"):
        info = arms[f"{family}_sparse"]["operator"]
        assert -1e-9 <= info["trace_error_exact"] <= info["trace_upper_bound"] + 1e-8
        assert info["spectral_error_to_full"] <= info["trace_error_exact"] + 1e-8
        assert (output / "graphs" / "group-7" / f"{family}_incidence.npz").exists()
        clustered = arms[f"{family}_cluster"]["operator"]
        assert clustered["spectral_error_to_full"] <= clustered["trace_error_exact"] + 1e-8
        assert (output / "graphs" / "group-7" / f"{family}_cluster_incidence.npz").exists()
    assert report["graphs"][0]["kernel_comparison"]["quantum_minus_markov_fro"] >= 0


def test_large_graph_path_avoids_dense_reference(tmp_path):
    config = _config(dense_limit=12, hard=False)
    config["cluster"]["enabled"] = False
    report, _ = _run(tmp_path, config)
    assert report["graphs"][0]["dense_reference_computed"] is False
    assert {row["method"] for row in report["records"]} == {
        "features_only", "graph_gcn", "quantum_sparse", "markov_sparse",
    }
    assert all(row["operator"]["trace_upper_bound"] >= 0
               for row in report["records"] if row["method"].endswith("sparse"))


def test_optional_graph_fusion_adds_matched_controls_and_selected_operators(tmp_path):
    config = _config(dense_limit=12, hard=False)
    config["cluster"]["enabled"] = False
    config["fusion"] = {"enabled": True, "alphas": [0.0, 0.5, 1.0]}
    report, output = _run(tmp_path, config)
    arms = {row["method"]: row for row in report["records"]}
    fused = {"graph_identity_fusion", "graph_quantum_sparse_fusion",
             "graph_markov_sparse_fusion"}
    assert set(arms) == {"features_only", "graph_gcn", "quantum_sparse",
                         "markov_sparse"} | fused
    for method in fused:
        scores = arms[method]["evaluation"]
        assert scores["selected_alpha"] in (0.0, 0.5, 1.0)
        assert [row["alpha"] for row in scores["validation_by_alpha"]] == [0.0, 0.5, 1.0]
        assert (output / "graphs" / "group-7" / f"{method}_theta.npz").exists()
    summary = report["summary"]["group"]
    assert summary["paired_fusion_quantum_minus_graph"]["n"] == 1
    assert summary["paired_fusion_quantum_minus_identity"]["n"] == 1
    assert summary["paired_fusion_quantum_minus_markov"]["n"] == 1


@pytest.mark.parametrize("alphas", [[0.0, 1.0], [0.0, 0.5, 0.5, 1.0],
                                      [0.0, 0.5, 1.1], [0.0, 0.5, 0.75]])
def test_fusion_config_requires_distinct_bounded_endpoints_and_interior(tmp_path, alphas):
    config = _config(dense_limit=12, hard=False)
    config["fusion"] = {"enabled": True, "alphas": alphas}
    with pytest.raises(ValueError, match="fusion.alphas"):
        _run(tmp_path, config)


def test_npz_mode_uses_explicit_held_out_indices(tmp_path):
    A, X, y, _ = make_synthetic_graph("pairwise", 3, n=24)
    split = stratified_node_split(y, seed=29)
    archive = tmp_path / "graph.npz"
    np.savez(archive, adjacency=A.toarray(), features=X, labels=y,
             train_idx=split["train"], validation_idx=split["validation"],
             test_idx=split["test"])
    config = _config(dense_limit=12, hard=False)
    config["cluster"]["enabled"] = False
    config["project"]["mode"] = "npz"
    config.pop("synthetic")
    config["npz"] = {"files": ["graph.npz"]}
    config["fusion"] = {"enabled": True, "alphas": [0.0, 0.5, 1.0]}
    report, _ = _run(tmp_path, config)
    assert report["graphs"][0]["metadata"]["source_sha256"]
    assert report["graphs"][0]["train_idx"] == split["train"].tolist()
    assert report["graphs"][0]["test_idx"] == split["test"].tolist()
    assert report["summary"]["npz"]["unit_type"] == "archive_independence_unverified"
    assert "graph_quantum_sparse_fusion" in {
        row["method"] for row in report["records"]}


def test_label_blind_trace_calibration_is_recorded(tmp_path):
    config = _config(dense_limit=32, hard=False)
    config["cluster"]["enabled"] = False
    config["kernel"].pop("times")
    config["kernel"].pop("weights")
    config["kernel"]["time_policy"] = "gcn_trace_match"
    config["kernel"]["time_candidates"] = [1.0, 3.0, 6.0]
    report, _ = _run(tmp_path, config)
    choice = report["graphs"][0]["time_selection"]
    target = choice["target_gcn_mean_diagonal"]
    values = choice["quantum_mean_diagonal_by_time"]
    selected = choice["time_candidates"].index(choice["selected_time"])
    assert abs(values[selected] - target) == min(abs(value - target) for value in values)
    assert choice["policy"] == "gcn_trace_match"


def test_npz_rejects_fractional_split_before_constructing(tmp_path):
    A, X, y, _ = make_synthetic_graph("pairwise", 3, n=24)
    split = stratified_node_split(y, seed=29)
    np.savez(tmp_path / "graph.npz", adjacency=A.toarray(), features=X, labels=y,
             train_idx=split["train"].astype(float) + 0.5,
             validation_idx=split["validation"], test_idx=split["test"])
    config = _config(dense_limit=12, hard=False)
    config["cluster"]["enabled"] = False
    config["project"]["mode"] = "npz"
    config.pop("synthetic")
    config["npz"] = {"files": ["graph.npz"]}
    with pytest.raises(ValueError, match="train indices must be"):
        _run(tmp_path, config)
