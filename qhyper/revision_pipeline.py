"""Executable, truth-blind P1 revision with separate search and feature audits.

The current-window binary projection is the only test observation. Training
hyperedges determine size quotas and scorer parameters; test hyperedges are
joined only after candidate pools and feature caches have been written. Every
feature variant scores one identical analysis pool. Cross-pool AP differences
are deliberately absent: search is assessed by coverage and oracle bounds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import yaml

from .candidate_proposal import propose_candidates
from .classic_features import classical_candidate_matrix, classical_feature_names
from .experiment_logger import RunProfiler, environment_record, sha256_file, write_json
from .graph_io import Edge, GraphRecord, load_cornell_event_files
from .hard_negatives import (match_hard_negatives_for_evaluation,
                             matched_average_precision)
from .hamiltonian import build_hamiltonian
from .metrics import candidate_recall, evaluate_candidate_predictions, evaluate_hyperedge_sets
from .real_data import split_events_into_windows
from .real_pipeline import (_archive_audit, _quantile_time_boundaries,
                            _resolve_local_files, load_real_protocol)
from .revision_candidates import propose_progressive_candidates
from .revision_selection import (observed_pairs, select_greedy_coverage,
                                 select_mdl_greedy, select_top_m)
from .revision_spectrum import (classical_boundary_gram_matrix,
                                matched_classical_spectral_feature_matrix,
                                quantum_group_feature_matrices)
from .scorer import fit_candidate_scorer, predict_candidate_scores, select_hyperedges
from .splits import validate_disjoint_graph_splits, validate_event_disjointness
from .synthetic import SyntheticConfig, generate_synthetic


@dataclass
class _Prepared:
    graph: GraphRecord
    candidates: list[Edge]
    pools: dict[int, dict[str, list[Edge]]]
    matrices: dict[str, np.ndarray]
    labels: np.ndarray | None
    timings: dict[str, float]


def load_revision_config(path: Path) -> dict[str, Any]:
    """Read a locked revision protocol before candidate or score computation."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("revision configuration must be a mapping")
    project = config.get("project", {})
    if (not isinstance(project, dict) or project.get("mode") not in {"real", "synthetic"}
            or isinstance(project.get("seed"), bool)
            or not isinstance(project.get("seed"), int)
            or not isinstance(project.get("protocol"), str)):
        raise ValueError("project requires mode, integer seed and protocol")
    candidate = config.get("candidate", {})
    budgets = candidate.get("budgets") if isinstance(candidate, dict) else None
    if (not isinstance(budgets, list) or not budgets
            or any(isinstance(b, bool) or not isinstance(b, int) or b < 1 for b in budgets)
            or budgets != sorted(set(budgets))):
        raise ValueError("candidate.budgets must be strictly increasing positive integers")
    if candidate.get("analysis_budget") not in budgets:
        raise ValueError("analysis_budget must belong to candidate.budgets")
    target_f1 = candidate.get("target_exact_f1")
    if (isinstance(target_f1, bool) or not isinstance(target_f1, (int, float))
            or not np.isfinite(target_f1) or not 0 < target_f1 <= 1):
        raise ValueError("candidate.target_exact_f1 must lie in (0,1]")
    if (not isinstance(candidate.get("min_size"), int)
            or not isinstance(candidate.get("max_size"), int)
            or not 2 <= candidate["min_size"] <= candidate["max_size"]):
        raise ValueError("candidate size range is invalid")
    for key in ("beam_width", "max_visited"):
        value = candidate.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"candidate.{key} must be positive integer")
    for key in ("search_exploration_fraction", "quota_exploration_fraction"):
        value = candidate.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
            raise ValueError(f"candidate.{key} must lie in [0,1)")
    times = config.get("time", {}).get("points")
    if (not isinstance(times, list) or not times
            or any(not isinstance(t, (int, float)) or isinstance(t, bool)
                   or not np.isfinite(t) or t < 0 for t in times)
            or len(set(times)) != len(times)):
        raise ValueError("time.points must contain distinct nonnegative finite values")
    spectral = config.get("spectral", {})
    if (not isinstance(spectral, dict) or not 0 < spectral.get("eps", -1) < 1
            or isinstance(spectral.get("dense_limit"), bool)
            or not isinstance(spectral.get("dense_limit"), int)
            or spectral["dense_limit"] < 2):
        raise ValueError("spectral requires eps in (0,1) and positive dense_limit")
    scorer = config.get("scorer", {})
    c_grid = scorer.get("C_grid") if isinstance(scorer, dict) else None
    if (not isinstance(c_grid, list) or not c_grid
            or any(not isinstance(c, (int, float)) or isinstance(c, bool)
                   or not np.isfinite(c) or c <= 0 for c in c_grid)):
        raise ValueError("scorer.C_grid must contain positive finite values")
    if scorer.get("class_weight", None) is not None:
        raise ValueError("revision scoring requires class_weight: null; probabilities are diagnostic")
    selection = config.get("selection", {})
    grid = selection.get("max_selected_grid") if isinstance(selection, dict) else None
    if (not isinstance(grid, list) or not grid
            or any(isinstance(m, bool) or not isinstance(m, int) or m < 1 for m in grid)):
        raise ValueError("selection.max_selected_grid must contain positive integers")
    if project["mode"] == "real" and not isinstance(config.get("real", {}).get("base_protocol"), str):
        raise ValueError("real mode requires real.base_protocol")
    if project["mode"] == "synthetic" and not isinstance(config.get("synthetic"), dict):
        raise ValueError("synthetic mode requires synthetic settings")
    return config


def _synthetic_sources(config: dict[str, Any]) -> tuple[dict[str, list[GraphRecord]], dict]:
    settings = config["synthetic"]
    roles = settings.get("split_seeds", {})
    if (set(roles) != {"train", "validation", "test"}
            or any(not isinstance(roles[role], list) or not roles[role] for role in roles)):
        raise ValueError("synthetic split_seeds need nonempty train/validation/test lists")
    all_seeds = [seed for values in roles.values() for seed in values]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("synthetic graph seeds must be disjoint across roles")
    cases = settings.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("synthetic.cases must be nonempty")
    graph_args = settings.get("graph", {})
    sources: dict[str, list[GraphRecord]] = {}
    for case in cases:
        specification = SyntheticConfig(case=str(case),
                                        n_nodes=int(graph_args["n_nodes"]),
                                        num_hyperedges=int(graph_args["num_hyperedges"]),
                                        edge_sizes=tuple(int(v) for v in graph_args["edge_sizes"]),
                                        n_blocks=int(graph_args["n_blocks"]))
        records = []
        for role in ("train", "validation", "test"):
            for seed in roles[role]:
                graph = generate_synthetic(specification, seed=int(seed)).graph
                graph.split = role
                records.append(graph)
        sources[f"synthetic/{case}"] = records
    return sources, {"mode": "synthetic", "cases": cases, "split_seeds": roles,
                     "graph": graph_args}


def _real_sources(config_path: Path, config: dict[str, Any]
                  ) -> tuple[dict[str, list[GraphRecord]], dict]:
    base_path = (config_path.parent / config["real"]["base_protocol"]).resolve()
    base = load_real_protocol(base_path)
    sources: dict[str, list[GraphRecord]] = {}
    provenance: dict[str, Any] = {"mode": "real", "base_protocol": str(base_path),
                                  "base_protocol_sha256": sha256_file(base_path),
                                  "sources": {}}
    for dataset in base["datasets"]:
        files = _resolve_local_files(base_path, dataset)
        manifest = _archive_audit(dataset, files)
        source = load_cornell_event_files(
            nverts_path=str(files["nverts"]), simplices_path=str(files["simplices"]),
            times_path=str(files["times"]), source_id=dataset["source_id"],
            source_url=dataset["source_url"],
            license_name=dataset.get("license_or_access_terms"), min_event_size=2)
        if (dataset.get("expected_raw_event_count") is not None and
                source.provenance["raw_event_count"] != dataset["expected_raw_event_count"]):
            raise ValueError("source raw event count differs from locked protocol")
        if (dataset.get("expected_num_nodes") is not None and
                source.num_nodes != dataset["expected_num_nodes"]):
            raise ValueError("source node count differs from locked protocol")
        boundaries = (dataset["window_boundaries"] if dataset.get("window_boundaries") is not None
                      else _quantile_time_boundaries(source, dataset["windowing"]["window_count"]))
        records = split_events_into_windows(source, boundaries=boundaries,
                                            labels=dataset["window_labels"])
        validate_disjoint_graph_splits({role: [graph.graph_id for graph in records
                                               if graph.split == role]
                                        for role in ("train", "validation", "test")})
        validate_event_disjointness({role: [event_id for graph in records
                                              if graph.split == role
                                              for event_id in graph.hyperedge_event_ids or []]
                                     for role in ("train", "validation", "test")})
        sources[dataset["source_id"]] = records
        provenance["sources"][dataset["source_id"]] = {
            "raw_file_sha256": {name: sha256_file(path) for name, path in files.items()},
            "archive_manifest_present": manifest is not None,
            "window_boundaries": boundaries,
            "window_labels": dataset["window_labels"],
            "raw_event_count": source.provenance["raw_event_count"],
        }
    return sources, provenance


def _train_size_counts(graphs: list[GraphRecord], min_k: int, max_k: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for graph in graphs:
        if graph.split != "train":
            continue
        for edge in set(graph.hyperedges_truth or []):
            if min_k <= len(edge) <= max_k:
                counts[len(edge)] += 1
    return counts


def _quota_for_budgets(counts: Counter[int], sizes: list[int], budgets: list[int],
                       exploration_fraction: float) -> dict[int, dict[int, int]]:
    """Sequential apportionment keeps each size's quota monotone in budget."""
    frequency = np.array([counts[k] for k in sizes], dtype=float)
    if frequency.sum() == 0:
        frequency = np.ones(len(sizes), dtype=float)
    weight = ((1 - exploration_fraction) * frequency / frequency.sum()
              + exploration_fraction / len(sizes))
    allocated = np.zeros(len(sizes), dtype=int)
    result = {}
    for position in range(1, max(budgets) + 1):
        deficits = position * weight - allocated
        chosen = int(np.argmax(deficits))
        allocated[chosen] += 1
        if position in budgets:
            result[position] = {k: int(v) for k, v in zip(sizes, allocated)}
    return result


def _graph_seed(base_seed: int, graph_id: str) -> int:
    value = sha256(f"{base_seed}:{graph_id}".encode("utf-8")).digest()
    return int.from_bytes(value[:4], "little")


def _prepare_graph(graph: GraphRecord, config: dict[str, Any], max_k: int,
                   quotas: dict[int, dict[int, int]], folder: Path) -> _Prepared:
    observed = graph.observed()
    A = observed.adjacency_csr
    candidate_cfg = config["candidate"]
    budgets = candidate_cfg["budgets"]
    seed = _graph_seed(config["project"]["seed"], graph.graph_id)
    pools: dict[int, dict[str, list[Edge]]] = {}
    audits: dict[str, Any] = {}
    timings: dict[str, float] = {}
    for budget in budgets:
        started = perf_counter()
        progress, audit = propose_progressive_candidates(
            A, budget, min_k=candidate_cfg["min_size"], max_k=max_k,
            seed=seed, per_size_budget=quotas[budget],
            beam_width=candidate_cfg["beam_width"],
            exploration_fraction=candidate_cfg["search_exploration_fraction"],
            max_visited=candidate_cfg["max_visited"])
        timings[f"progressive_B{budget}"] = perf_counter() - started
        started = perf_counter()
        legacy = propose_candidates(A, budget=budget,
                                    min_k=candidate_cfg["min_size"], max_k=max_k,
                                    seed=seed, per_size_budget=quotas[budget])
        timings[f"legacy_B{budget}"] = perf_counter() - started
        pools[budget] = {"progressive": progress, "legacy": legacy}
        audits[str(budget)] = audit
    analysis = pools[candidate_cfg["analysis_budget"]]["progressive"]
    folder.mkdir(parents=True, exist_ok=True)
    # Both files are intentionally written before this function reads truth.
    write_json(folder / "candidate_pools_label_free.json", {
        "graph_id": graph.graph_id,
        "budgets": {str(b): {method: [list(e) for e in edges]
                             for method, edges in pools[b].items()} for b in budgets},
        "progressive_search_audit": audits,
        "size_quotas_from_training": {str(b): {str(k): v for k, v in quotas[b].items()}
                                       for b in budgets},
    })
    H = build_hamiltonian(A, kind="norm_laplacian")
    started = perf_counter()
    c0 = classical_candidate_matrix(A, analysis)
    timings["C0"] = perf_counter() - started
    started = perf_counter()
    cgram, cgram_names = classical_boundary_gram_matrix(H, analysis)
    timings["Cgram"] = perf_counter() - started
    started = perf_counter()
    quantum, quantum_names = quantum_group_feature_matrices(
        H, analysis, config["time"]["points"], eps=config["spectral"]["eps"],
        dense_limit=config["spectral"]["dense_limit"])
    timings["Q3_Qspec_shared"] = perf_counter() - started
    matched: dict[str, np.ndarray] = {}
    matched_names: dict[str, tuple[str, ...]] = {}
    for kind in ("heat", "wave"):
        started = perf_counter()
        matched[kind], matched_names[kind] = matched_classical_spectral_feature_matrix(
            H, analysis, config["time"]["points"], kind=kind,
            eps=config["spectral"]["eps"], dense_limit=config["spectral"]["dense_limit"])
        timings["Cmatched_" + kind] = perf_counter() - started
    base = np.hstack((c0, cgram))
    matrices = {
        "C0_original": c0,
        "C0_Cgram": base,
        "C0_Cgram_Q3": np.hstack((base, quantum["Q3"])),
        "C0_Cgram_Qspec": np.hstack((base, quantum["Qspec"])),
        "C0_Cgram_Q3_Qspec": np.hstack((base, quantum["Q3"], quantum["Qspec"])),
        "C0_Cgram_Cmatched_heat": np.hstack((base, matched["heat"])),
        "C0_Cgram_Cmatched_wave": np.hstack((base, matched["wave"])),
    }
    np.savez_compressed(folder / "feature_cache_label_free.npz", **matrices)
    write_json(folder / "feature_manifest_label_free.json", {
        "graph_id": graph.graph_id,
        "analysis_budget": candidate_cfg["analysis_budget"],
        "candidate_count": len(analysis),
        "feature_names": {
            "C0_original": classical_feature_names(),
            "Cgram": cgram_names,
            "Q3": quantum_names["Q3"],
            "Qspec": quantum_names["Qspec"],
            "Cmatched_heat": matched_names["heat"],
            "Cmatched_wave": matched_names["wave"],
        },
        "cache_contains_truth": False,
    })
    return _Prepared(graph, analysis, pools, matrices, None, timings)


def _attach_evaluation_labels(prepared: _Prepared) -> None:
    truth = set(prepared.graph.hyperedges_truth or [])
    prepared.labels = np.asarray([int(edge in truth) for edge in prepared.candidates], dtype=int)


def _stack(items: list[_Prepared], role: str, variant: str
           ) -> tuple[np.ndarray, np.ndarray, list[_Prepared]]:
    selected = [item for item in items if item.graph.split == role]
    if not selected:
        raise ValueError(f"source has no {role} graphs")
    return (np.vstack([item.matrices[variant] for item in selected]),
            np.concatenate([item.labels for item in selected if item.labels is not None]),
            selected)


def _threshold_for_exact_f1(validation: list[_Prepared], scores: np.ndarray) -> dict[str, float]:
    """Maximize *full* exact-set F1 on validation, including unproposed truth."""
    labels = np.concatenate([item.labels for item in validation if item.labels is not None])
    if len(labels) != len(scores) or not len(scores):
        raise ValueError("validation labels and scores must be nonempty and aligned")
    total_true = sum(len(set(item.graph.hyperedges_truth or [])) for item in validation)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = fp = 0
    best_f1 = 0.0
    best_threshold = float(np.nextafter(np.max(scores), np.inf))
    index = 0
    while index < len(scores):
        threshold = float(sorted_scores[index])
        while index < len(scores) and sorted_scores[index] == threshold:
            if sorted_labels[index]:
                tp += 1
            else:
                fp += 1
            index += 1
        fn = total_true - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
        if f1 > best_f1 + 1e-12:
            best_f1, best_threshold = float(f1), threshold
    return {"threshold": best_threshold, "validation_full_f1": best_f1}


def _micro_exact(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    values = list(rows)
    tp = sum(int(value["tp"]) for value in values)
    fp = sum(int(value["fp"]) for value in values)
    fn = sum(int(value["fn"]) for value in values)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "jaccard": tp / (tp + fp + fn) if tp + fp + fn else 0.0}


def _member_identity(graph: GraphRecord, edge: Edge) -> frozenset[Any]:
    """Compare exact member sets across windows with source, not local, IDs."""
    return frozenset(graph.original_node_ids[node] for node in edge)


def _test_truth_strata(item: _Prepared,
                       train_members: set[frozenset[Any]] | None
                       ) -> dict[str, set[Edge] | None]:
    truth = set(item.graph.hyperedges_truth or [])
    by_size = {size: [set(edge) for edge in truth if len(edge) == size]
               for size in {len(edge) for edge in truth}}
    nested = {edge for edge in truth
              if any(set(edge) < larger for size, groups in by_size.items()
                     if size > len(edge) for larger in groups)}
    # Independently generated synthetic graphs reuse integer node indices but
    # those indices do not denote the same entities across graphs.  An unseen
    # member-set statistic is meaningful only for persistent real source IDs.
    unseen = (None if train_members is None else
              {edge for edge in truth
               if _member_identity(item.graph, edge) not in train_members})
    return {"nested": nested, "train_unseen": unseen}


def _stratified_recall(selected: Iterable[Edge],
                       strata: dict[str, set[Edge] | None]
                       ) -> dict[str, dict[str, float | int | None] | None]:
    chosen = set(selected)
    return {name: (None if truth is None else
                   {"truth_count": len(truth),
                    "covered_count": len(chosen & truth),
                    "recall": len(chosen & truth) / len(truth) if truth else None})
            for name, truth in strata.items()}


def _projection_error(A: Any, selected: Iterable[Edge]) -> dict[str, float | int]:
    """Observed dyad discrepancy; selected hyperedges are still evaluated separately."""
    observed = observed_pairs(A)
    projected = set().union(*(set(combinations(edge, 2)) for edge in selected))
    false_negative = len(observed - projected)
    false_positive = len(projected - observed)
    return {"observed_dyads": len(observed),
            "uncovered_dyads": false_negative,
            "spurious_dyads": false_positive,
            "dyad_error_count": false_negative + false_positive,
            "dyad_error_rate": ((false_negative + false_positive) / len(observed)
                                if observed else float(false_positive > 0))}


def _select_unsupervised_limit(validation: list[_Prepared], method: str,
                               grid: list[int]) -> dict[str, float | int]:
    best: tuple[float, int] | None = None
    for limit in sorted(set(grid)):
        metrics = []
        for item in validation:
            selector = select_greedy_coverage if method == "coverage" else select_mdl_greedy
            selected = selector(item.graph.observed().adjacency_csr, item.candidates,
                                max_selected=limit)
            metrics.append(evaluate_hyperedge_sets(selected, item.graph.hyperedges_truth or []))
        f1 = float(_micro_exact(metrics)["f1"])
        if best is None or f1 > best[0] + 1e-12 or (abs(f1 - best[0]) <= 1e-12 and limit < best[1]):
            best = (f1, limit)
    assert best is not None
    return {"max_selected": best[1], "validation_full_f1": best[0]}


def _select_top_m_limit(validation: list[_Prepared], scores: np.ndarray,
                        grid: list[int]) -> dict[str, float | int]:
    if len(scores) != sum(len(item.candidates) for item in validation):
        raise ValueError("top-M validation scores are not aligned")
    best: tuple[float, int] | None = None
    for limit in sorted(set(grid)):
        offset = 0
        metrics = []
        for item in validation:
            local = scores[offset:offset + len(item.candidates)]
            offset += len(item.candidates)
            selected = select_top_m(item.candidates, local, limit)
            metrics.append(evaluate_hyperedge_sets(selected, item.graph.hyperedges_truth or []))
        f1 = float(_micro_exact(metrics)["f1"])
        if best is None or f1 > best[0] + 1e-12 or (abs(f1 - best[0]) <= 1e-12 and limit < best[1]):
            best = (f1, limit)
    assert best is not None
    return {"max_selected": best[1], "validation_full_f1": best[0]}


def _coverage_report(items: list[_Prepared], budgets: list[int],
                     effective_max: int, target_f1: float,
                     strata_by_graph: dict[str, dict[str, set[Edge] | None]],
                     min_size: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for budget in budgets:
        output[str(budget)] = {}
        for method in ("progressive", "legacy"):
            by_graph = {}
            total_covered = total_truth = total_candidates = eligible_truth = 0
            by_size: dict[str, dict[str, int]] = {}
            nested_truth = nested_covered = unseen_truth = unseen_covered = 0
            for item in items:
                if item.graph.split != "test":
                    continue
                metric = candidate_recall(item.pools[budget][method],
                                          item.graph.hyperedges_truth or [])
                by_graph[item.graph.graph_id] = metric
                strata = _stratified_recall(item.pools[budget][method],
                                            strata_by_graph[item.graph.graph_id])
                by_graph[item.graph.graph_id]["truth_strata"] = strata
                nested_truth += strata["nested"]["truth_count"]
                nested_covered += strata["nested"]["covered_count"]
                if strata["train_unseen"] is not None:
                    unseen_truth += strata["train_unseen"]["truth_count"]
                    unseen_covered += strata["train_unseen"]["covered_count"]
                total_covered += metric["covered_count"]
                total_truth += metric["true_count"]
                total_candidates += metric["candidate_count"]
                eligible_truth += sum(min_size <= len(edge) <= effective_max
                                      for edge in set(item.graph.hyperedges_truth or []))
                for size, values in metric["by_cardinality"].items():
                    entry = by_size.setdefault(size, {"true_count": 0, "covered_count": 0})
                    entry["true_count"] += values["true_count"]
                    entry["covered_count"] += values["covered_count"]
            recall = total_covered / total_truth if total_truth else None
            output[str(budget)][method] = {
                "by_graph": by_graph,
                "total_candidates": total_candidates,
                "covered_count": total_covered,
                "truth_count": total_truth,
                "candidate_recall": recall,
                "by_cardinality": {
                    size: {**values, "recall": values["covered_count"] / values["true_count"]}
                    for size, values in by_size.items()},
                "k_ge_3": {
                    "truth_count": sum(values["true_count"] for size, values in by_size.items()
                                       if int(size) >= 3),
                    "covered_count": sum(values["covered_count"] for size, values in by_size.items()
                                         if int(size) >= 3),
                    "recall": (sum(values["covered_count"] for size, values in by_size.items()
                                   if int(size) >= 3) /
                               sum(values["true_count"] for size, values in by_size.items()
                                   if int(size) >= 3)
                               if any(int(size) >= 3 for size in by_size) else None),
                },
                "truth_strata": {
                    "nested": {"truth_count": nested_truth, "covered_count": nested_covered,
                               "recall": nested_covered / nested_truth if nested_truth else None},
                    "train_unseen": (
                        {"truth_count": unseen_truth,
                         "covered_count": unseen_covered,
                         "recall": unseen_covered / unseen_truth if unseen_truth else None}
                        if all(strata_by_graph[item.graph.graph_id]["train_unseen"] is not None
                               for item in items if item.graph.split == "test") else None),
                },
                "size_scope_recall_ceiling": (eligible_truth / total_truth
                                               if total_truth else None),
                "capacity_recall_ceiling": (min(1.0, total_candidates / total_truth)
                                             if total_truth else None),
                "capacity_recall_ceiling_note": (
                    "loose cross-graph slot-count bound; ignores per-graph allocation "
                    "and candidate size eligibility"),
                "oracle_exact_f1_bound": (2 * recall / (1 + recall)
                                          if recall is not None else None),
                "target_exact_f1": target_f1,
                "minimum_recall_for_target_f1": target_f1 / (2 - target_f1),
                "target_f1_possible_in_this_pool": (recall >= target_f1 / (2 - target_f1)
                                                    if recall is not None else None),
            }
    return output


def _run_source(source_id: str, graphs: list[GraphRecord], config: dict[str, Any],
                folder: Path) -> dict[str, Any]:
    if not all(graph.observation_mode == "binary" for graph in graphs):
        raise ValueError("revision primary protocol requires binary graph observations")
    candidate_cfg = config["candidate"]
    train_truth = [edge for graph in graphs if graph.split == "train"
                   for edge in set(graph.hyperedges_truth or [])]
    if not train_truth:
        raise ValueError("each source requires training hyperedges")
    effective_max = min(candidate_cfg["max_size"], max(map(len, train_truth)),
                        min(graph.num_nodes for graph in graphs))
    if effective_max < candidate_cfg["min_size"]:
        raise ValueError("training-derived maximum size is below candidate minimum")
    sizes = list(range(candidate_cfg["min_size"], effective_max + 1))
    counts = _train_size_counts(graphs, sizes[0], sizes[-1])
    quotas = _quota_for_budgets(counts, sizes, candidate_cfg["budgets"],
                                candidate_cfg["quota_exploration_fraction"])
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "training_only_quota_policy.json", {
        "source_id": source_id, "training_truth_size_counts": dict(counts),
        "effective_max_size": effective_max, "budgets": candidate_cfg["budgets"],
        "quota_formula": "w_k=(1-eta)*train_count_k/sum(train_count)+eta/number_of_sizes; sequential largest deficit at each budget step",
        "eta": candidate_cfg["quota_exploration_fraction"],
        "quotas": {str(b): {str(k): v for k, v in quotas[b].items()}
                   for b in candidate_cfg["budgets"]},
        "depends_on_test_truth": False,
    })
    prepared = []
    for graph in graphs:
        graph_folder = folder / "graphs" / sha256(graph.graph_id.encode()).hexdigest()[:16]
        prepared.append(_prepare_graph(graph, config, effective_max, quotas, graph_folder))
    # The label-free files above are fixed before this join.
    for item in prepared:
        _attach_evaluation_labels(item)
    train_members = ({_member_identity(graph, edge)
                      for graph in graphs if graph.split == "train"
                      for edge in set(graph.hyperedges_truth or [])}
                     if config["project"]["mode"] == "real" else None)
    variants = list(prepared[0].matrices)
    val_items = [item for item in prepared if item.graph.split == "validation"]
    test_items = [item for item in prepared if item.graph.split == "test"]
    test_strata = {item.graph.graph_id: _test_truth_strata(item, train_members)
                   for item in test_items}
    report: dict[str, Any] = {
        "source_id": source_id,
        "effective_max_size_from_training": effective_max,
        "size_scope_note": "test truth below min_size or above effective_max_size remains in the denominator",
        "train_unseen_basis": ("persistent source node IDs across real time windows"
                               if train_members is not None else
                               "not_applicable: independent synthetic graphs have no shared node identities"),
        "candidate_coverage_by_budget": _coverage_report(
            prepared, candidate_cfg["budgets"], effective_max,
            float(candidate_cfg["target_exact_f1"]), test_strata,
            candidate_cfg["min_size"]),
        "selection_without_candidate_scores": {},
        "feature_variants_common_pool": {},
        "feature_comparisons": {},
        "timing_seconds_by_graph": {item.graph.graph_id: item.timings for item in prepared},
        "search_timing_note": "Each budget invokes each search independently; times are standalone run costs, not cumulative increments.",
    }
    matchings = {}
    degree_index = classical_feature_names().index("degree_mean")
    for item in test_items:
        matchings[item.graph.graph_id] = match_hard_negatives_for_evaluation(
            item.labels,
            [len(edge) for edge in item.candidates],
            item.matrices["C0_original"][:, degree_index],
            ["progressive"] * len(item.candidates),
            seed=_graph_seed(config["project"]["seed"], item.graph.graph_id))
    report["size_degree_matched_clique_negatives_for_evaluation"] = {
        "matching_note": ("All candidates have the same progressive source. Size is a hard "
                          "constraint; degree quantile bin is attempted and can relax "
                          "to size-only. This does not match candidate provenance."),
        "by_graph": matchings,
    }
    for method in ("coverage", "mdl"):
        choice = _select_unsupervised_limit(
            val_items, method, config["selection"]["max_selected_grid"])
        per_graph = {}
        for item in test_items:
            selector = select_greedy_coverage if method == "coverage" else select_mdl_greedy
            selected = selector(item.graph.observed().adjacency_csr, item.candidates,
                                max_selected=int(choice["max_selected"]))
            per_graph[item.graph.graph_id] = evaluate_hyperedge_sets(
                selected, item.graph.hyperedges_truth or [])
            per_graph[item.graph.graph_id]["projection_error"] = _projection_error(
                item.graph.observed().adjacency_csr, selected)
            per_graph[item.graph.graph_id]["truth_strata"] = _stratified_recall(
                selected, test_strata[item.graph.graph_id])
        report["selection_without_candidate_scores"][method] = {
            "validation_choice": choice, "test_by_graph": per_graph,
            "test_micro_exact": _micro_exact(per_graph.values()),
            "interpretation": ("pure monotone coverage baseline" if method == "coverage"
                               else "two-part MDL heuristic, no true-hypergraph guarantee"),
        }
    predictions = []
    for variant in variants:
        train_x, train_y, _ = _stack(prepared, "train", variant)
        val_x, val_y, _ = _stack(prepared, "validation", variant)
        test_x, test_y, _ = _stack(prepared, "test", variant)
        if len(np.unique(train_y)) != 2:
            raise ValueError(f"{source_id}: {variant} training pool lacks both classes")
        best_model = best_scores = None
        best_c = None
        best_ap = -np.inf
        started = perf_counter()
        for c in sorted(set(config["scorer"]["C_grid"])):
            model = fit_candidate_scorer(
                train_x, train_y, C=float(c), class_weight=None,
                random_state=config["project"]["seed"])
            val_scores = predict_candidate_scores(model, val_x)
            val_metric = evaluate_candidate_predictions(val_y, val_scores)
            quality = (val_metric["ap"] if val_metric["ap"] is not None
                       else -val_metric["log_loss"])
            if quality > best_ap + 1e-12:
                best_model, best_scores, best_c, best_ap = model, val_scores, c, quality
        assert best_model is not None and best_scores is not None
        threshold = _threshold_for_exact_f1(val_items, best_scores)
        top_m = _select_top_m_limit(
            val_items, best_scores, config["selection"]["max_selected_grid"])
        test_scores = predict_candidate_scores(best_model, test_x)
        elapsed = perf_counter() - started
        sizes_test = [len(edge) for item in test_items for edge in item.candidates]
        pooled = evaluate_candidate_predictions(test_y, test_scores, sizes=sizes_test)
        higher_order_mask = np.asarray(sizes_test) >= 3
        higher_order = evaluate_candidate_predictions(
            test_y[higher_order_mask], test_scores[higher_order_mask],
            sizes=np.asarray(sizes_test)[higher_order_mask])
        offset = 0
        per_graph = {}
        top_m_by_graph = {}
        for item in test_items:
            local_scores = test_scores[offset:offset + len(item.candidates)]
            offset += len(item.candidates)
            selected = select_hyperedges(item.candidates, local_scores,
                                         float(threshold["threshold"]))
            per_graph[item.graph.graph_id] = {
                "candidate": evaluate_candidate_predictions(
                    item.labels, local_scores, sizes=[len(edge) for edge in item.candidates]),
                "exact": evaluate_hyperedge_sets(selected, item.graph.hyperedges_truth or []),
                "projection_error": _projection_error(
                    item.graph.observed().adjacency_csr, selected),
                "truth_strata": _stratified_recall(
                    selected, test_strata[item.graph.graph_id]),
                "size_degree_matched_clique_diagnostic": matched_average_precision(
                    item.labels, local_scores, matchings[item.graph.graph_id]),
            }
            predictions.extend({
                "variant": variant, "graph_id": item.graph.graph_id,
                "node_ids": list(edge), "label_for_evaluation": int(label),
                "score": float(score),
            } for edge, label, score in zip(item.candidates, item.labels, local_scores))
            top_m_selected = select_top_m(item.candidates, local_scores,
                                          int(top_m["max_selected"]))
            top_m_by_graph[item.graph.graph_id] = evaluate_hyperedge_sets(
                top_m_selected, item.graph.hyperedges_truth or [])
            top_m_by_graph[item.graph.graph_id]["projection_error"] = _projection_error(
                item.graph.observed().adjacency_csr, top_m_selected)
            top_m_by_graph[item.graph.graph_id]["truth_strata"] = _stratified_recall(
                top_m_selected, test_strata[item.graph.graph_id])
        report["feature_variants_common_pool"][variant] = {
            "feature_columns": int(train_x.shape[1]),
            "training_rows": int(len(train_y)), "validation_rows": int(len(val_y)),
            "test_rows": int(len(test_y)),
            "selected_C_on_validation": best_c,
            "validation_selection_metric": best_ap,
            "validation_exact_threshold": threshold,
            "validation_top_m": top_m,
            "test_candidate_pooled": pooled,
            "test_candidate_k_ge_3": higher_order,
            "test_exact_by_graph": per_graph,
            "test_exact_micro": _micro_exact(row["exact"] for row in per_graph.values()),
            "test_top_m_by_graph": top_m_by_graph,
            "test_top_m_micro": _micro_exact(top_m_by_graph.values()),
            "fit_and_score_seconds": elapsed,
        }
    base_ap = report["feature_variants_common_pool"]["C0_Cgram"]["test_candidate_pooled"]["ap"]
    for name in ("C0_Cgram_Q3", "C0_Cgram_Qspec", "C0_Cgram_Q3_Qspec",
                 "C0_Cgram_Cmatched_heat", "C0_Cgram_Cmatched_wave"):
        value = report["feature_variants_common_pool"][name]["test_candidate_pooled"]["ap"]
        report["feature_comparisons"][name + "_vs_C0_Cgram"] = (
            None if value is None or base_ap is None else float(value - base_ap))
    with (folder / "test_predictions_for_evaluation.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(folder / "revision_source_report.json", report)
    return report


def run_revision(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run a complete exploratory v2 audit without changing old P1 outputs."""
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    config = load_revision_config(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {output_dir}")
    profiler = RunProfiler()
    with profiler.stage("load_sources"):
        if config["project"]["mode"] == "real":
            sources, provenance = _real_sources(config_path, config)
        else:
            sources, provenance = _synthetic_sources(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(output_dir / "environment.json", environment_record())
    write_json(output_dir / "input_provenance.json", {
        "revision_config": str(config_path),
        "revision_config_sha256": sha256_file(config_path),
        **provenance,
    })
    repository = Path(__file__).resolve().parents[1]
    source_files = ["main.py"] + sorted(
        str(path.relative_to(repository)) for path in (repository / "qhyper").glob("*.py"))
    write_json(output_dir / "code_provenance.json", {
        "repository_directory": str(repository),
        "sha256_by_path": {name: sha256_file(repository / name) for name in source_files},
        "hashes_include_uncommitted_edits": True,
    })
    reports = {}
    for index, (source_id, graphs) in enumerate(sources.items()):
        with profiler.stage("source:" + source_id):
            reports[source_id] = _run_source(source_id, graphs, config,
                                             output_dir / "sources" / f"source-{index}")
    summary = {
        "protocol": config["project"]["protocol"],
        "mode": config["project"]["mode"],
        "gate": {"status": "exploratory_not_confirmatory",
                 "reason": "revised hypotheses require new independent held-out sources or periods"},
        "analysis_budget": config["candidate"]["analysis_budget"],
        "sources": reports,
    }
    write_json(output_dir / "revision_summary.json", summary)
    write_json(output_dir / "runtime_memory.json", {
        **profiler.record(),
        "rss_scope": "load_sources and whole-source stages only; feature-stage RSS is not sampled separately",
        "feature_stage_timing_location": "sources/*/revision_source_report.json:timing_seconds_by_graph",
    })
    return summary
