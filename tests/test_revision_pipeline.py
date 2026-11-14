"""End-to-end and leakage checks for the exploratory revision runner."""

import json
from pathlib import Path

import numpy as np
import yaml

from qhyper.graph_io import GraphRecord
from qhyper.projection import project_hypergraph
from qhyper.revision_pipeline import (_Prepared, _coverage_report, _prepare_graph,
                                      _quota_for_budgets, _real_sources,
                                      _test_truth_strata, load_revision_config,
                                      run_revision)


ROOT = Path(__file__).resolve().parents[1]


def test_revision_smoke_runs_on_common_pool_and_records_limits(tmp_path):
    config = ROOT / "configs" / "p1_revision_smoke.yaml"
    result = run_revision(config, tmp_path / "result")
    assert result["gate"]["status"] == "exploratory_not_confirmatory"
    assert set(result["sources"]) == {"synthetic/S1", "synthetic/S2"}
    for report in result["sources"].values():
        variants = report["feature_variants_common_pool"]
        assert {"C0_original", "C0_Cgram", "C0_Cgram_Q3", "C0_Cgram_Qspec",
                "C0_Cgram_Cmatched_heat", "C0_Cgram_Cmatched_wave"} <= set(variants)
        assert len({entry["test_rows"] for entry in variants.values()}) == 1
        assert all("test_exact_micro" in entry and "test_top_m_micro" in entry
                   for entry in variants.values())
        assert "independent" in report["search_timing_note"]
        assert report["train_unseen_basis"].startswith("not_applicable")
        assert "size_degree_matched_clique_negatives_for_evaluation" in report
        for by_method in report["candidate_coverage_by_budget"].values():
            for entry in by_method.values():
                recall = entry["candidate_recall"]
                assert np.isclose(entry["oracle_exact_f1_bound"], 2 * recall / (1 + recall))
                assert entry["candidate_recall"] <= entry["size_scope_recall_ceiling"] + 1e-12
                assert "k_ge_3" in entry
                assert entry["truth_strata"]["train_unseen"] is None
                assert "loose cross-graph" in entry["capacity_recall_ceiling_note"]
        for entry in report["feature_variants_common_pool"].values():
            for graph in entry["test_exact_by_graph"].values():
                assert graph["projection_error"]["spurious_dyads"] == 0
                assert "nested" in graph["truth_strata"]
    cache = next((tmp_path / "result").rglob("feature_cache_label_free.npz"))
    with np.load(cache) as payload:
        assert "labels" not in payload.files
    manifest = next((tmp_path / "result").rglob("feature_manifest_label_free.json"))
    assert json.loads(manifest.read_text())["cache_contains_truth"] is False
    runtime = json.loads((tmp_path / "result" / "runtime_memory.json").read_text())
    assert runtime["rss_peak_is_sampled"] is True
    assert "whole-source" in runtime["rss_scope"]
    code = json.loads((tmp_path / "result" / "code_provenance.json").read_text())
    expected_code = {"main.py"} | {
        str(path.relative_to(ROOT)) for path in (ROOT / "qhyper").glob("*.py")}
    assert set(code["sha256_by_path"]) == expected_code


def test_size_scope_counts_both_lower_and_upper_bounds_and_real_identity():
    truth = [(0, 1), (0, 1, 2), (0, 1, 2, 3)]
    A = project_hypergraph(truth, 4, mode="binary")
    graph = GraphRecord("test", 4, A, hyperedges_truth=truth, split="test",
                        original_node_ids=(10, 11, 12, 13))
    item = _Prepared(graph, [(0, 1, 2)],
                     {1: {"progressive": [(0, 1, 2)],
                          "legacy": [(0, 1, 2)]}}, {}, None, {})
    strata = _test_truth_strata(item, {frozenset((10, 11))})
    assert strata["train_unseen"] == {(0, 1, 2), (0, 1, 2, 3)}
    result = _coverage_report([item], [1], effective_max=3,
                              target_f1=0.5,
                              strata_by_graph={graph.graph_id: strata}, min_size=3)
    entry = result["1"]["progressive"]
    assert entry["size_scope_recall_ceiling"] == 1 / 3
    assert entry["truth_strata"]["train_unseen"]["truth_count"] == 2


def test_revision_caches_do_not_depend_on_hidden_test_truth(tmp_path):
    config = load_revision_config(ROOT / "configs" / "p1_revision_smoke.yaml")
    config["candidate"]["budgets"] = [4]
    config["candidate"]["analysis_budget"] = 4
    config["time"]["points"] = [0.25]
    A = project_hypergraph([(0, 1, 2)], 3, mode="binary")
    records = [GraphRecord("same-observation", 3, A,
                           hyperedges_truth=truth, split="test")
               for truth in ([(0, 1, 2)], [(0, 1), (0, 2), (1, 2)])]
    quotas = {4: {2: 3, 3: 1}}
    first = _prepare_graph(records[0], config, 3, quotas, tmp_path / "a")
    second = _prepare_graph(records[1], config, 3, quotas, tmp_path / "b")
    assert first.candidates == second.candidates
    assert first.pools == second.pools
    assert first.labels is second.labels is None
    assert all(np.array_equal(first.matrices[name], second.matrices[name])
               for name in first.matrices)
    assert ((tmp_path / "a" / "candidate_pools_label_free.json").read_bytes() ==
            (tmp_path / "b" / "candidate_pools_label_free.json").read_bytes())


def test_train_size_quota_is_monotone_and_has_exploration():
    from collections import Counter

    allocation = _quota_for_budgets(Counter({2: 80, 3: 20, 4: 0}),
                                    [2, 3, 4], [10, 40], 0.3)
    assert sum(allocation[10].values()) == 10
    assert sum(allocation[40].values()) == 40
    assert all(allocation[40][k] >= allocation[10][k] for k in (2, 3, 4))
    assert allocation[40][4] > 0


def test_revision_real_loader_reuses_locked_chronological_protocol(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    base = yaml.safe_load((ROOT / "configs" / "p1_real_pilot.yaml").read_text())
    for index, dataset in enumerate(base["datasets"]):
        folder = tmp_path / f"source-{index}"
        folder.mkdir()
        (folder / "nverts.txt").write_text("3\n" * 5)
        (folder / "simplices.txt").write_text("0\n1\n2\n" * 5)
        (folder / "times.txt").write_text("0\n1\n2\n3\n4\n")
        dataset["source_id"] = f"toy/source-{index}/temporal"
        dataset["source_url"] = f"https://example.org/source-{index}"
        dataset["expected_raw_event_count"] = 5
        dataset["expected_num_nodes"] = 3
        dataset["require_archive_manifest"] = False
        dataset["files"] = {key: f"../source-{index}/{key}.txt"
                            for key in ("nverts", "simplices", "times")}
        dataset.pop("archive_url", None)
        dataset["window_boundaries"] = [0.5, 1.5, 2.5, 3.5]
        dataset.pop("windowing", None)
        dataset["window_labels"] = ["train", "train", "validation", "test", "test"]
    (config_dir / "toy_base.yaml").write_text(yaml.safe_dump(base))
    revision = {"real": {"base_protocol": "toy_base.yaml"}}
    sources, provenance = _real_sources(config_dir / "revision.yaml", revision)
    assert len(sources) == 2
    assert all([graph.split for graph in graphs] ==
               ["train", "train", "validation", "test", "test"]
               for graphs in sources.values())
    assert all(len(record["raw_file_sha256"]) == 3
               for record in provenance["sources"].values())
