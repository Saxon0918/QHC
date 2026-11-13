from dataclasses import replace
import json

import numpy as np
import pytest
import yaml

from qhyper.experiment_logger import RunProfiler
from qhyper.p1_pipeline import load_protocol, prepare_graph
from qhyper.synthetic import SyntheticConfig, generate_synthetic


def test_protocol_rejects_undeclared_feature_and_scorer_changes(tmp_path):
    config = {
        "project": {"seed": 42},
        "scenarios": {"S1": {"n_nodes": 12, "num_hyperedges": 8,
                             "edge_sizes": [3], "n_blocks": 3}},
        "split_seeds": {"train": [1], "validation": [2], "test": [3]},
        "observation": {"projection": "binary", "allow_node_features": False,
                        "use_test_hyperedges_as_input": False},
        "hamiltonian": {"kind": "norm_laplacian", "scale": 0.5,
                        "feature_laplacian_weight": 0.0},
        "time": {"points": [0.25]},
        "candidate": {"budget": 12, "min_size": 3, "max_size": 3,
                      "generators": ["local_clique"],
                      "enforce_clique_in_exact_projection": True},
        "quantum": {"incoherent_reference": "exact"},
        "scorer": {"model": "logistic_regression", "threshold_metric": "f1"},
        "reporting": {"primary_metric": "average_precision",
                      "precision_at_budget": 3},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert load_protocol(path)["candidate"]["budget"] == 12
    config["scenarios"]["S1"]["feature_signal"] = 1.0
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="feature"):
        load_protocol(path)
    config["scenarios"]["S1"].pop("feature_signal")
    config["scorer"]["model"] = "mlp"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="logistic"):
        load_protocol(path)


def test_feature_cache_is_invariant_to_hidden_truth(tmp_path):
    sample = generate_synthetic(SyntheticConfig(
        case="S1", n_nodes=12, num_hyperedges=8,
        edge_sizes=(3,), n_blocks=3), seed=17)
    graph = sample.graph
    alternative = replace(graph, hyperedges_truth=[], hyperedge_event_ids=None)
    config = {
        "project": {"seed": 42},
        "candidate": {"budget": 12, "min_size": 3, "max_size": 3,
                      "generators": ["local_clique", "maximal_clique_subsets"]},
        "time": {"points": [0.25]},
        "quantum": {"incoherent_reference": "exact", "phase_samples": 2,
                    "max_state_columns": 16},
    }
    first = prepare_graph(graph, config, tmp_path / "first", RunProfiler())
    second = prepare_graph(alternative, config, tmp_path / "second", RunProfiler())
    assert first.candidates == second.candidates
    assert not np.array_equal(first.labels, second.labels)
    for variant in first.features:
        assert np.array_equal(first.features[variant], second.features[variant])
    first_payload = json.loads((tmp_path / "first" /
                                (graph.graph_id + "_candidates.json")).read_text())
    assert all("label_if_training" not in row for row in first_payload)
    with np.load(tmp_path / "first" / (graph.graph_id + ".npz")) as saved:
        assert not any("truth" in key or "label" in key for key in saved.files)
