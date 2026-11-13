"""End-to-end check that temporal source files reach the shared P1 evaluator."""

import json
from pathlib import Path

import yaml
import pytest

from qhyper.real_pipeline import load_real_protocol, run_real_p1
from qhyper.real_sweep import run_real_seed_sweep
from qhyper.synthetic import SyntheticConfig, generate_synthetic


def _write_temporal_source(folder, *, seed_offset):
    folder.mkdir()
    sizes, nodes, times = [], [], []
    for window in range(5):
        graph = generate_synthetic(SyntheticConfig(
            case="S1", n_nodes=14, num_hyperedges=10,
            edge_sizes=(3, 4), n_blocks=3),
            seed=seed_offset + window).graph
        for edge in graph.hyperedges_truth:
            sizes.append(str(len(edge)))
            nodes.extend(str(node) for node in edge)
            times.append(str(window))
    for name, values in (("nverts", sizes), ("simplices", nodes), ("times", times)):
        (folder / (name + ".txt")).write_text("\n".join(values) + "\n", encoding="utf-8")


def _real_config():
    datasets = []
    for name in ("source-a", "source-b"):
        datasets.append({
            "source_id": "test/" + name + "/temporal",
            "source_url": "https://example.org/" + name,
            "files": {key: "../" + name + "/" + key + ".txt"
                      for key in ("nverts", "simplices", "times")},
            "window_boundaries": [0.5, 1.5, 2.5, 3.5],
            "window_labels": ["train", "train", "validation", "test", "test"],
        })
    return {
        "project": {"name": "test", "protocol": "temporal_real_p1_test", "seed": 42},
        "minimum_meaningful_delta_ap": 0.02,
        "datasets": datasets,
        "observation": {"projection": "binary", "allow_node_features": False,
                        "use_test_hyperedges_as_input": False},
        "hamiltonian": {"kind": "norm_laplacian", "scale": 0.5,
                        "feature_laplacian_weight": 0.0},
        "time": {"points": [0.5]},
        "candidate": {"budget": 30, "min_size": 2, "max_size": 4,
                      "generators": ["local_clique", "maximal_clique_subsets"],
                      "enforce_clique_in_exact_projection": True},
        "quantum": {"incoherent_reference": "exact", "phase_samples": 2,
                    "max_state_columns": 32},
        "scorer": {"model": "logistic_regression", "C_grid": [0.1, 1.0],
                   "class_weight": "balanced", "threshold_metric": "f1"},
        "reporting": {"primary_metric": "average_precision",
                      "precision_at_budget": 5, "bootstrap_replicates": 20},
    }


def test_temporal_real_pipeline_uses_two_source_clusters(tmp_path):
    for index, name in enumerate(("source-a", "source-b")):
        _write_temporal_source(tmp_path / name, seed_offset=50 + 20 * index)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "real.yaml"
    config_path.write_text(yaml.safe_dump(_real_config()), encoding="utf-8")
    assert len(load_real_protocol(config_path)["datasets"]) == 2

    output = tmp_path / "results"
    report = run_real_p1(config_path, output)
    locked = load_real_protocol(output / "config_resolved.yaml")
    assert all(len(dataset["window_boundaries"]) == 4
               and "windowing" not in dataset
               and Path(dataset["files"]["times"]).is_absolute()
               for dataset in locked["datasets"])
    audit = json.loads((output / "real_input_audit.json").read_text(encoding="utf-8"))
    assert report["primary_analysis_unit"] == "source_dataset"
    assert report["gate_1"]["status"] == "not_evaluated"
    assert set(report["sources"]) == {
        "test/source-a/temporal", "test/source-b/temporal"}
    assert len(audit["datasets"]) == 2
    assert all(len(entry["window_summary"]) == 5 for entry in audit["datasets"])
    manifest = json.loads((output / "data_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["source_ids"]) == 2
    for entry in report["sources"].values():
        source_manifest = json.loads((Path(entry["output"]) /
                                      "data_manifest.json").read_text(encoding="utf-8"))
        assert len(source_manifest["graph_records"]) == 5
        assert all(graph["source_files"] for graph in source_manifest["graph_records"])
        assert entry["planned_comparison_details"]["Q3_vs_C0_classical"]["ci_low"] is None


def test_seed_sweep_keeps_the_same_time_windows(tmp_path):
    for index, name in enumerate(("source-a", "source-b")):
        _write_temporal_source(tmp_path / name, seed_offset=100 + 20 * index)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = _real_config()
    config["replicate_seeds"] = [42, 43, 44]
    config_path = config_dir / "real.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    sweep = run_real_seed_sweep(config_path, tmp_path / "sweep")
    assert sweep["seed_list"] == [42, 43, 44]
    assert sweep["locked_window_boundaries"] == ((0.5, 1.5, 2.5, 3.5),) * 2
    assert all(len(entry["planned_comparisons"]["Q3_vs_C0_classical"]
                   ["delta_ap_by_seed"]) == 3 for entry in sweep["sources"].values())
    assert sweep["gate_1"]["status"] == "not_evaluated"


def test_real_pipeline_rejects_mislabeled_source_release(tmp_path):
    for index, name in enumerate(("source-a", "source-b")):
        _write_temporal_source(tmp_path / name, seed_offset=150 + 20 * index)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = _real_config()
    config["datasets"][0]["expected_raw_event_count"] = 999999
    config_path = config_dir / "real.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="raw event count does not match"):
        run_real_p1(config_path, tmp_path / "results")
