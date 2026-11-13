"""Cross-graph synthetic P1 discrimination with a shared candidate pool.

Feature construction receives only ``GraphRecord.observed()``. Truth is joined
after truth-free feature caches are complete, solely for supervised training
on train graphs, validation selection, and held-out evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import yaml
from scipy import sparse

from .candidate_proposal import candidate_coverage, propose_candidates, recall_at_budget
from .classic_features import (classical_candidate_matrix,
                               classical_feature_names, dynamic_family_matrix)
from .experiment_logger import RunProfiler, environment_record, sha256_file, write_json
from .graph_io import CandidateRecord, GraphRecord
from .group_response import extract_group_quantum_features
from .hard_negatives import (match_hard_negatives_for_evaluation,
                             matched_average_precision)
from .hamiltonian import build_hamiltonian
from .metrics import (canonical_edge_set, evaluate_candidate_predictions,
                      evaluate_hyperedge_sets, paired_bootstrap_delta)
from .scorer import (choose_validation_threshold, fit_candidate_scorer,
                     predict_candidate_scores, select_hyperedges)
from .splits import validate_disjoint_graph_splits
from .synthetic import SyntheticConfig, generate_synthetic, triangle_ambiguity_example


def load_protocol(path: Path) -> dict[str, Any]:
    """Load and check the fixed topology-only, binary P1 protocol."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    required = ("project", "scenarios", "split_seeds", "observation",
                "hamiltonian", "time", "candidate", "quantum", "scorer")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("missing configuration sections: {}".format(missing))
    if config["observation"].get("projection") != "binary":
        raise ValueError("P1 primary protocol requires binary projection")
    if config["observation"].get("use_test_hyperedges_as_input") is not False:
        raise ValueError("test hyperedges cannot be model inputs")
    if config["observation"].get("allow_node_features") is not False:
        raise ValueError("P1 primary protocol is topology-only; feature ablations are separate")
    if config["hamiltonian"].get("kind") != "norm_laplacian":
        raise ValueError("P1 primary Hamiltonian must be the normalized Laplacian")
    if float(config["hamiltonian"].get("scale", 0.0)) != 0.5:
        raise ValueError("P1 primary Hamiltonian uses the fixed 0.5 scale")
    if float(config["hamiltonian"].get("feature_laplacian_weight", -1)) != 0.0:
        raise ValueError("P1 primary Hamiltonian cannot use node features")
    if set(config["split_seeds"]) != {"train", "validation", "test"}:
        raise ValueError("split_seeds must specify train, validation and test")
    seeds = {split: list(values) for split, values in config["split_seeds"].items()}
    if any(not values for values in seeds.values()):
        raise ValueError("each split needs at least one independent graph seed")
    if any(not isinstance(seed, int) for values in seeds.values() for seed in values):
        raise ValueError("graph seeds must be integers")
    if len({seed for values in seeds.values() for seed in values}) != sum(map(len, seeds.values())):
        raise ValueError("a seed cannot appear in multiple graph splits")
    scenarios = config["scenarios"]
    if not scenarios or any(name not in {"S1", "S2"} for name in scenarios):
        raise ValueError("P1 synthetic scenarios must be S1 and/or S2")
    for name, settings in scenarios.items():
        SyntheticConfig(case=name, **settings)
        if float(settings.get("feature_signal", 0.0)) != 0.0:
            raise ValueError("node features require a separate feature ablation protocol")
        if settings.get("edge_add_prob", 0) or settings.get("edge_drop_prob", 0):
            raise ValueError("noisy observations require a separate non-clique protocol")
    times = config["time"].get("points")
    if not isinstance(times, list) or not times or any(float(t) <= 0 for t in times):
        raise ValueError("time.points must contain fixed positive values")
    candidates = config["candidate"]
    if candidates.get("budget", 0) < 1 or candidates.get("min_size", 0) < 2:
        raise ValueError("candidate budget and minimum size are invalid")
    if candidates.get("max_size", 0) < candidates["min_size"]:
        raise ValueError("candidate max_size is below min_size")
    if candidates.get("enforce_clique_in_exact_projection") is not True:
        raise ValueError("noiseless projection requires clique candidates in P1")
    if not candidates.get("generators") or set(candidates["generators"]) - {
        "local_clique", "maximal_clique_subsets"}:
        raise ValueError("candidate generators must be explicit supported channels")
    if config["quantum"].get("incoherent_reference") not in {"exact", "rademacher"}:
        raise ValueError("unknown noncoherent reference")
    if config["scorer"].get("model") != "logistic_regression":
        raise ValueError("P1 only implements the shared logistic_regression scorer")
    if config["scorer"].get("threshold_metric") != "f1":
        raise ValueError("P1 validation threshold is selected by candidate F1")
    reporting = config.get("reporting", {})
    if reporting.get("primary_metric") != "average_precision":
        raise ValueError("P1 primary metric is average_precision")
    if not isinstance(reporting.get("precision_at_budget"), int) or reporting["precision_at_budget"] < 1:
        raise ValueError("reporting.precision_at_budget must be a positive integer")
    return config


def _hash_observation(adjacency: sparse.csr_matrix) -> str:
    matrix = sparse.csr_matrix(adjacency)
    digest = hashlib.sha256()
    for array in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def _hash_truth(edges: Iterable[Iterable[int]]) -> str:
    payload = json.dumps(sorted(canonical_edge_set(edges)), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class PreparedGraph:
    graph: GraphRecord
    records: list[CandidateRecord]
    features: dict[str, np.ndarray]
    labels: np.ndarray
    coverage: dict[str, Any]
    budget_curve: list[dict[str, Any]]

    @property
    def candidates(self) -> list[tuple[int, ...]]:
        return [record.node_ids for record in self.records]


def make_synthetic_graphs(config: Mapping[str, Any]) -> list[GraphRecord]:
    """Draw independent graphs, preserving raw truth only for later labels."""
    graphs = []
    ids: dict[str, list[str]] = {part: [] for part in ("train", "validation", "test")}
    for scenario, settings in config["scenarios"].items():
        specification = SyntheticConfig(case=scenario, **settings)
        for split, seeds in config["split_seeds"].items():
            for seed in seeds:
                graph = generate_synthetic(specification, seed=int(seed)).graph
                graph.split = split
                graphs.append(graph)
                ids[split].append(graph.graph_id)
    validate_disjoint_graph_splits(ids)
    return graphs


def _variant_matrices(
    z_classic: np.ndarray, z_quantum: np.ndarray,
    controls: Mapping[str, np.ndarray], phase_reference: np.ndarray,
) -> dict[str, np.ndarray]:
    feature_names = classical_feature_names()
    c2_index = feature_names.index("short_time_c2_coefficient")
    coupling_index = feature_names.index("external_coupling_inner_product_sum")
    # These two columns encode the same exact second-order statistic up to k.
    without_explicit_c2 = np.delete(z_classic, [coupling_index, c2_index], axis=1)
    return {
        "C0_classical": z_classic,
        "C0_without_explicit_c2": without_explicit_c2,
        "C1_heat": np.hstack((z_classic, controls["heat"])),
        "C2_wave": np.hstack((z_classic, controls["wave"])),
        "C3_spectral": np.hstack((z_classic, controls["spectral"])),
        "C4_short_time_only": z_classic[:, c2_index:c2_index + 1],
        "Q1_pairwise": np.hstack((z_classic, controls["quantum_pair"])),
        "Q2_coherent": np.hstack((z_classic, z_quantum[:, 0::3])),
        "Q2_incoherent": np.hstack((z_classic, z_quantum[:, 1::3])),
        "Q2_contrast": np.hstack((z_classic, z_quantum[:, 2::3])),
        "Q3_group": np.hstack((z_classic, z_quantum)),
        "Q3_independent_columns": np.hstack((z_classic, z_quantum[:, 0::3],
                                               z_quantum[:, 1::3])),
        "Q4_random_phase": np.hstack((z_classic, phase_reference)),
    }


def prepare_graph(
    graph: GraphRecord, config: Mapping[str, Any], cache_dir: Path,
    profiler: RunProfiler,
) -> PreparedGraph:
    """Build and cache features from observation only, then attach truth labels."""
    observation = graph.observed()
    assert observation.hyperedges_truth is None
    graph_id = observation.graph_id
    candidate_cfg = config["candidate"]
    with profiler.stage(graph_id + ":candidate_proposal"):
        records = propose_candidates(
            observation.adjacency_csr, observation.features,
            budget=int(candidate_cfg["budget"]),
            min_k=int(candidate_cfg["min_size"]),
            max_k=int(candidate_cfg["max_size"]),
            seed=int(config["project"]["seed"]),
            generators=candidate_cfg["generators"],
            per_size_budget=candidate_cfg.get("per_size_budget"),
            graph_id=graph_id, return_records=True,
        )
    if not records:
        raise RuntimeError("no candidates for {}; inspect graph and candidate settings".format(graph_id))
    candidates = [record.node_ids for record in records]
    sources = [record.proposal_sources for record in records]
    times = [float(t) for t in config["time"]["points"]]
    h = build_hamiltonian(observation.adjacency_csr, kind="norm_laplacian", lam=0.0)

    with profiler.stage(graph_id + ":classical_features"):
        z_classic = classical_candidate_matrix(
            observation.adjacency_csr, candidates, observation.features, sources=sources)
    quantum_cfg = config["quantum"]
    exact_reference = quantum_cfg["incoherent_reference"] == "exact"
    with profiler.stage(graph_id + ":quantum_group_features"):
        z_quantum, quantum_diagnostics = extract_group_quantum_features(
            h, candidates, times, exact_incoherent=exact_reference,
            n_phase_samples=int(quantum_cfg["phase_samples"]),
            seed=int(config["project"]["seed"]),
            max_state_columns=int(quantum_cfg["max_state_columns"]),
        )
    controls: dict[str, np.ndarray] = {}
    for family in ("heat", "wave", "spectral", "quantum_pair"):
        with profiler.stage(graph_id + ":" + family + "_features"):
            controls[family] = dynamic_family_matrix(
                observation.adjacency_csr, h, candidates, times, family,
                max_state_columns=int(quantum_cfg["max_state_columns"]),
            )
    with profiler.stage(graph_id + ":random_phase_control"):
        _, phase_diagnostics = extract_group_quantum_features(
            h, candidates, times, exact_incoherent=False,
            n_phase_samples=int(quantum_cfg["phase_samples"]),
            seed=int(config["project"]["seed"]),
            max_state_columns=int(quantum_cfg["max_state_columns"]),
        )
        # Use only the random-phase mean as the Q4 dephased control. Sampling
        # variance is logged by the extractor, but can itself encode coherence.
        phase_reference = phase_diagnostics["R_inc"]
    variants = _variant_matrices(z_classic, z_quantum, controls, phase_reference)
    if any(matrix.shape[0] != len(records) or not np.isfinite(matrix).all()
           for matrix in variants.values()):
        raise FloatingPointError("feature matrix is not finite or candidate-aligned")

    # The cache is deliberately written before any reference to test truth.
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / (graph_id + ".npz"), **variants,
                        R_coh=quantum_diagnostics["R_coh"],
                        R_inc=quantum_diagnostics["R_inc"],
                        coherence_contrast=quantum_diagnostics["C"],
                        phase_sample_variance=phase_diagnostics["phase_sample_variance"],
                        phase_estimator_variance=phase_diagnostics["phase_estimator_variance"])
    write_json(cache_dir / (graph_id + "_candidates.json"),
               [record.feature_payload() for record in records])

    truth = graph.hyperedges_truth
    if truth is None:
        raise ValueError("P1 synthetic evaluation requires hidden ground truth")
    truth_set = canonical_edge_set(truth)
    labels = np.asarray([int(edge in truth_set) for edge in candidates], dtype=int)
    coverage = candidate_coverage(candidates, truth)
    eligible = sum(int(candidate_cfg["min_size"] <= len(edge) <= candidate_cfg["max_size"])
                   for edge in truth_set)
    coverage["size_scope_recall_ceiling"] = (eligible / len(truth_set)
                                              if truth_set else None)
    coverage["truth_excluded_by_size"] = len(truth_set) - eligible
    budgets = sorted(set([0, min(10, len(candidates)),
                          min(50, len(candidates)), len(candidates)]))
    curve = recall_at_budget(candidates, truth, budgets)
    return PreparedGraph(graph, records, variants, labels, coverage, curve)


def _stack(prepared: Sequence[PreparedGraph], split: str,
           variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    selected = [item for item in prepared if item.graph.split == split]
    if not selected:
        raise ValueError("empty {} split".format(split))
    matrix = np.vstack([item.features[variant] for item in selected])
    labels = np.concatenate([item.labels for item in selected])
    groups = np.concatenate([
        np.repeat(item.graph.graph_id, len(item.records)) for item in selected
    ])
    edges = [record.node_ids for item in selected for record in item.records]
    return matrix, labels, groups, edges


def run_p1(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run the declared P0/P1 synthetic protocol and save all raw outputs."""
    config_path, output_dir = Path(config_path), Path(output_dir)
    config = load_protocol(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty run directory: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(output_dir / "environment.json", environment_record())
    profiler = RunProfiler()
    with profiler.stage("synthetic_generation"):
        graphs = make_synthetic_graphs(config)
    first, second = triangle_ambiguity_example()
    ambiguity_ok = (first.adjacency_csr != second.adjacency_csr).nnz == 0
    if not ambiguity_ok or canonical_edge_set(first.hyperedges_truth or []) == canonical_edge_set(second.hyperedges_truth or []):
        raise AssertionError("S3 projection ambiguity witness failed")
    write_json(output_dir / "s3_ambiguity.json", {
        "same_binary_projection": True,
        "truth_world_1": first.hyperedges_truth,
        "truth_world_2": second.hyperedges_truth,
        "interpretation": "不能仅从 A 区分两个真实超图",
    })

    prepared = [prepare_graph(graph, config, output_dir / "feature_cache", profiler)
                for graph in graphs]
    coverage = {item.graph.graph_id: {"split": item.graph.split,
                                      "coverage": item.coverage,
                                      "recall_at_budget": item.budget_curve}
                for item in prepared}
    write_json(output_dir / "candidate_coverage.json", coverage)
    plan = Path(__file__).resolve().parents[1] / "docs" / "RESEARCH_PLAN.md"
    manifest = {
        "protocol": config["project"]["protocol"],
        "config_sha256": sha256_file(config_path),
        "research_plan_sha256": sha256_file(plan) if plan.exists() else None,
        "graph_records": [
            {"graph_id": graph.graph_id, "split": graph.split,
             "scenario": graph.provenance.get("dataset"),
             "parameters": graph.provenance.get("parameters"),
             "observation_sha256": _hash_observation(graph.adjacency_csr),
             "truth_sha256_for_audit_only": _hash_truth(graph.hyperedges_truth or []),
             "truth_visible_to_feature_builder": False}
            for graph in graphs
        ],
    }
    write_json(output_dir / "data_manifest.json", manifest)

    # A compact truth-free diagnostic table; truth labels occur only in final
    # evaluation files and never in feature_cache or feature_diagnostics.csv.
    with (output_dir / "feature_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        first_time = float(config["time"]["points"][0])
        writer.writerow(["graph_id", "candidate_id", "size",
                         "R_coh_t{:g}".format(first_time),
                         "R_inc_t{:g}".format(first_time),
                         "C_t{:g}".format(first_time),
                         "short_time_c2_coefficient"])
        c2_index = classical_feature_names().index("short_time_c2_coefficient")
        for item in prepared:
            zc = item.features["C0_classical"]
            zq = item.features["Q3_group"][:, zc.shape[1]:]
            for i, record in enumerate(item.records):
                writer.writerow([item.graph.graph_id, record.candidate_id, record.size,
                                 zq[i, 0], zq[i, 1], zq[i, 2], zc[i, c2_index]])

    scorer_config = config["scorer"]
    variant_names = list(prepared[0].features)
    report: dict[str, Any] = {"protocol": config["project"]["protocol"],
                              "primary_metric": "average_precision",
                              "variants": {}, "planned_comparisons": {}}
    test_scores: dict[str, np.ndarray] = {}
    test_labels = _stack(prepared, "test", variant_names[0])[1]
    test_groups = _stack(prepared, "test", variant_names[0])[2]
    all_predictions: list[dict[str, Any]] = []

    for variant in variant_names:
        train_x, train_y, _, _ = _stack(prepared, "train", variant)
        val_x, val_y, _, _ = _stack(prepared, "validation", variant)
        test_x, _, _, _ = _stack(prepared, "test", variant)
        c_grid = [float(value) for value in scorer_config.get(
            "C_grid", [scorer_config.get("C", 1.0)])]
        if not c_grid or any(value <= 0 or not np.isfinite(value) for value in c_grid):
            raise ValueError("scorer C_grid must contain positive finite values")
        selected_model = None
        selected_validation_scores = None
        selected_c = None
        selected_validation_ap = -np.inf
        with profiler.stage("scorer_fit:" + variant):
            for value in sorted(set(c_grid)):
                candidate_model = fit_candidate_scorer(
                    train_x, train_y, C=value,
                    class_weight=scorer_config.get("class_weight", "balanced"),
                    random_state=int(config["project"]["seed"]),
                )
                candidate_scores = predict_candidate_scores(candidate_model, val_x)
                candidate_ap = evaluate_candidate_predictions(
                    val_y, candidate_scores)["ap"]
                if candidate_ap is None:
                    raise ValueError("validation split has no positives for AP model selection")
                if candidate_ap > selected_validation_ap:
                    selected_model = candidate_model
                    selected_validation_scores = candidate_scores
                    selected_c = value
                    selected_validation_ap = candidate_ap
        assert selected_model is not None and selected_validation_scores is not None
        model = selected_model
        validation_scores = selected_validation_scores
        threshold = choose_validation_threshold(val_y, validation_scores)
        scores = predict_candidate_scores(model, test_x)
        test_scores[variant] = scores
        pooled = evaluate_candidate_predictions(
            test_labels, scores,
            sizes=[record.size for item in prepared if item.graph.split == "test"
                   for record in item.records],
        )
        offset = 0
        by_graph = {}
        for item in prepared:
            if item.graph.split != "test":
                continue
            local_scores = scores[offset:offset + len(item.records)]
            offset += len(item.records)
            selected = select_hyperedges(item.candidates, local_scores,
                                         float(threshold["threshold"]))
            by_graph[item.graph.graph_id] = {
                "candidate": evaluate_candidate_predictions(
                    item.labels, local_scores,
                    sizes=[record.size for record in item.records],
                    budget=int(config.get("reporting", {}).get("precision_at_budget", 10))),
                "exact_reconstruction_diagnostic": evaluate_hyperedge_sets(
                    selected, item.graph.hyperedges_truth or []),
                "candidate_recall_upper_bound": item.coverage["recall"],
            }
            for record, label, score in zip(item.records, item.labels, local_scores):
                all_predictions.append({
                    "variant": variant, "graph_id": item.graph.graph_id,
                    "candidate_id": record.candidate_id, "node_ids": record.node_ids,
                    "label_for_evaluation": int(label), "score": float(score),
                })
        per_graph_precision = [entry["candidate"]["precision_at_budget"]
                               for entry in by_graph.values()
                               if entry["candidate"]["precision_at_budget"] is not None]
        report["variants"][variant] = {
            "feature_count": int(train_x.shape[1]),
            "training_candidates": int(train_x.shape[0]),
            "validation_candidates": int(val_x.shape[0]),
            "test_candidates": int(test_x.shape[0]),
            "validation_threshold": threshold,
            "validation_selected_C": selected_c,
            "validation_ap": selected_validation_ap,
            "test_candidate_metrics_pooled": pooled,
            "test_precision_at_budget_mean_across_graphs": (
                float(np.mean(per_graph_precision)) if per_graph_precision else None),
            "test_by_graph": by_graph,
        }

    # Predeclared size/degree/source matching is an extra held-out diagnostic.
    # It is computed only after every variant has completed fitting, validation
    # selection and test scoring, and never changes the primary full-pool AP.
    degree_index = classical_feature_names().index("degree_mean")
    hard_matches: dict[str, dict[str, Any]] = {}
    for item in prepared:
        if item.graph.split != "test":
            continue
        hard_matches[item.graph.graph_id] = match_hard_negatives_for_evaluation(
            item.labels,
            [record.size for record in item.records],
            item.features["C0_classical"][:, degree_index],
            [record.proposal_sources for record in item.records],
            seed=int(config["project"]["seed"]),
        )
    write_json(output_dir / "hard_negative_matching.json", hard_matches)
    for variant in variant_names:
        scores = test_scores[variant]
        offset = 0
        matched_aps = []
        for item in prepared:
            if item.graph.split != "test":
                continue
            local_scores = scores[offset:offset + len(item.records)]
            offset += len(item.records)
            matched = matched_average_precision(
                item.labels, local_scores, hard_matches[item.graph.graph_id])
            report["variants"][variant]["test_by_graph"][item.graph.graph_id][
                "matched_false_clique_diagnostic"] = matched
            if matched["matched_ap"] is not None:
                matched_aps.append(matched["matched_ap"])
        report["variants"][variant]["test_matched_ap_mean_across_graphs"] = (
            float(np.mean(matched_aps)) if matched_aps else None)

    baseline = test_scores["Q3_group"]
    for comparator in ("C0_classical", "C2_wave", "C3_spectral"):
        report["planned_comparisons"]["Q3_vs_" + comparator] = paired_bootstrap_delta(
            test_labels, test_scores[comparator], baseline, test_groups,
            n_bootstrap=int(config.get("reporting", {}).get("bootstrap_replicates", 1000)),
            seed=int(config["project"]["seed"]), metric="ap",
        )
    report["gate_1"] = {
        "status": "not_evaluated",
        "reason": "S1/S2 synthetic data are only the first P1 check; two real-data protocols and prespecified effect size are still required",
    }
    write_json(output_dir / "candidate_metrics.json", report)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    expected_variants = set(variant_names)
    aligned = all(set(item.features) == expected_variants and
                  all(matrix.shape[0] == len(item.records)
                      for matrix in item.features.values()) for item in prepared)
    cache_label_free = True
    for item in prepared:
        graph_id = item.graph.graph_id
        with np.load(output_dir / "feature_cache" / (graph_id + ".npz")) as saved:
            cache_label_free &= not any("label" in key.lower() or "truth" in key.lower()
                                        for key in saved.files)
        cached_candidates = json.loads((output_dir / "feature_cache" /
                                        (graph_id + "_candidates.json")).read_text(
                                            encoding="utf-8"))
        cache_label_free &= all("label_if_training" not in record and
                                "prediction_score" not in record
                                for record in cached_candidates)
    split_ids = {split: [item.graph.graph_id for item in prepared
                         if item.graph.split == split]
                 for split in ("train", "validation", "test")}
    validate_disjoint_graph_splits(split_ids)
    if not aligned or not cache_label_free:
        raise AssertionError("candidate alignment or truth-free cache check failed")
    write_json(output_dir / "fairness_checks.json", {
        "candidate_rows_aligned_across_variants": aligned,
        "feature_cache_has_no_truth_or_label_fields": cache_label_free,
        "graph_ids_by_split": split_ids,
        "shared_observation_source": "one GraphRecord.observed() per graph for all variants",
        "shared_scorer_family": "StandardScaler + LogisticRegression",
        "same_validation_C_search_grid": config["scorer"].get(
            "C_grid", [config["scorer"].get("C", 1.0)]),
        "threshold_selection_input": "validation candidate labels and scores only",
        "C0_exact_short_time_feature_column": classical_feature_names().index(
            "short_time_c2_coefficient"),
        "candidate_budget_per_graph": int(config["candidate"]["budget"]),
        "precision_at_budget_per_graph": int(config["reporting"]["precision_at_budget"]),
        "time_grid": config["time"]["points"],
    })
    write_json(output_dir / "runtime_memory.json", profiler.record())
    return report
