"""Prespecified temporal real-data P1 candidate discrimination protocol.

Raw Cornell event files are loaded locally. Each train, validation and test
window has its own binary projection of complete events; test truth reaches
the shared P1 evaluator only after observation-only feature caches are made.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .experiment_logger import RunProfiler, environment_record, sha256_file, write_json
from .graph_io import GraphRecord, load_cornell_event_files
from .p1_pipeline import run_p1_graphs, validate_shared_p1_protocol
from .real_data import split_events_into_windows
from .splits import validate_disjoint_graph_splits, validate_event_disjointness


def load_real_protocol(path: Path) -> dict[str, Any]:
    """Load the fixed, topology-only, temporal P1 protocol and local file paths."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    validate_shared_p1_protocol(config)
    if not isinstance(config["project"].get("seed"), int) or not config["project"].get("protocol"):
        raise ValueError("project.seed and project.protocol are required")
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or len(datasets) < 2:
        raise ValueError("at least two prespecified temporal real datasets are required")
    seen_sources: set[str] = set()
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("each dataset must be a mapping")
        source_id = dataset.get("source_id")
        if (not isinstance(source_id, str) or len(source_id.split("/")) < 3 or
                source_id in seen_sources):
            raise ValueError("datasets need distinct version-qualified source_id values")
        seen_sources.add(source_id)
        if not isinstance(dataset.get("source_url"), str) or not dataset["source_url"].startswith("https://"):
            raise ValueError("each dataset requires an HTTPS release page")
        for key in ("expected_raw_event_count", "expected_num_nodes"):
            value = dataset.get(key)
            if value is not None and (isinstance(value, bool) or
                                      not isinstance(value, int) or value < 1):
                raise ValueError(f"{key} must be a positive integer when specified")
        files = dataset.get("files")
        if not isinstance(files, dict) or set(files) != {"nverts", "simplices", "times"}:
            raise ValueError("temporal source files must specify nverts, simplices and times")
        for key, value in files.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"invalid {key} file path")
        labels = dataset.get("window_labels")
        boundaries = dataset.get("window_boundaries")
        windowing = dataset.get("windowing")
        if (boundaries is None) == (windowing is None):
            raise ValueError("specify exactly one of window_boundaries or windowing")
        if boundaries is not None:
            if not isinstance(boundaries, list) or len(boundaries) < 4:
                raise ValueError("at least five nonempty chronological windows are required")
            if (any(isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t)
                    for t in boundaries) or
                    any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
                raise ValueError("window_boundaries must be strictly increasing finite times")
            window_count = len(boundaries) + 1
        else:
            if not isinstance(windowing, dict) or windowing.get("method") != "unique_timestamp_quantiles":
                raise ValueError("windowing requires unique_timestamp_quantiles method")
            window_count = windowing.get("window_count")
            if isinstance(window_count, bool) or not isinstance(window_count, int) or window_count < 5:
                raise ValueError("window_count must be an integer of at least five")
        if not isinstance(labels, list) or len(labels) != window_count:
            raise ValueError("window_labels must match the number of windows")
        role_order = {"train": 0, "validation": 1, "test": 2}
        if any(role not in role_order for role in labels):
            raise ValueError("window roles must be train, validation or test")
        if labels != sorted(labels, key=role_order.get):
            raise ValueError("window roles must occur in train, validation, test order")
        if any(labels.count(role) < 1 for role in ("train", "validation", "test")):
            raise ValueError("each source needs train, validation and test windows")
        if labels.count("test") < 2:
            raise ValueError("each source needs at least two test time blocks")
    minimum_effect = config.get("minimum_meaningful_delta_ap")
    if (isinstance(minimum_effect, bool) or not isinstance(minimum_effect, (int, float))
            or not 0 < minimum_effect < 1):
        raise ValueError("minimum_meaningful_delta_ap must be fixed in (0,1)")
    return config


def _quantile_time_boundaries(source: GraphRecord, window_count: int) -> list[float]:
    """Choose equal-unique-timestamp windows without consulting memberships or scores."""
    if source.hyperedge_timestamps is None:
        raise ValueError("temporal source has no timestamps")
    times = np.asarray(source.hyperedge_timestamps, dtype=float)
    if not np.isfinite(times).all():
        raise ValueError("temporal source has nonfinite timestamps")
    unique = np.unique(times)
    if unique.size < window_count:
        raise ValueError("fewer distinct timestamps than requested windows")
    ranks = [int(np.ceil(i * unique.size / window_count))
             for i in range(1, window_count)]
    return [float(unique[rank]) for rank in ranks]


def _resolve_local_files(config_path: Path, dataset: dict[str, Any]) -> dict[str, Path]:
    files = {key: (config_path.parent / value).resolve()
             for key, value in dataset["files"].items()}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing local source files: " + ", ".join(missing))
    return files


def _archive_audit(dataset: dict[str, Any], files: dict[str, Path]) -> Any:
    manifest_path = files["nverts"].parent / "archive_manifest.json"
    if not manifest_path.is_file():
        if dataset.get("require_archive_manifest", False):
            raise FileNotFoundError(f"missing archive provenance manifest: {manifest_path}")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_id") != dataset["source_id"] or manifest.get("source_url") != dataset["source_url"]:
        raise ValueError("archive manifest source identity does not match protocol")
    if dataset.get("archive_url") and manifest.get("archive_url") != dataset["archive_url"]:
        raise ValueError("archive manifest URL does not match protocol")
    for key, path in files.items():
        if manifest.get("files", {}).get(key, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"archive manifest hash mismatch for {path}")
    return manifest


def _original_memberships(graph: GraphRecord) -> set[tuple[Any, ...]]:
    return {tuple(sorted(graph.original_node_ids[node] for node in edge))
            for edge in graph.hyperedges_truth or []}


def _source_audit(source: GraphRecord, windows: list[GraphRecord],
                  archive_manifest: Any) -> dict[str, Any]:
    by_split = {role: [window for window in windows if window.split == role]
                for role in ("train", "validation", "test")}
    member_sets = {
        role: set().union(*(_original_memberships(window) for window in parts))
        for role, parts in by_split.items()
    }
    node_sets = {role: set().union(*(set(window.original_node_ids) for window in parts))
                 for role, parts in by_split.items()}
    return {
        "source_id": source.provenance["source_id"],
        "source_url": source.provenance["source_url"],
        "license_or_access_terms": source.provenance.get("license"),
        "source_files": source.provenance["files"],
        "archive_manifest": archive_manifest,
        "raw_event_count": source.provenance["raw_event_count"],
        "raw_node_count_in_event_files": source.num_nodes,
        "retained_event_count": len(source.hyperedges_truth or []),
        "unique_member_set_count": len(_original_memberships(source)),
        "filtered_singleton_event_count": source.provenance["excluded_small_event_count"],
        "window_summary": [
            {"graph_id": graph.graph_id, "split": graph.split,
             "num_nodes": graph.num_nodes,
             "event_count": len(graph.hyperedges_truth or []),
             "unique_member_set_count": len(_original_memberships(graph)),
             "time_window": graph.provenance["window"]}
            for graph in windows
        ],
        "repeated_memberships_across_roles": {
            "train_validation": len(member_sets["train"] & member_sets["validation"]),
            "train_test": len(member_sets["train"] & member_sets["test"]),
            "validation_test": len(member_sets["validation"] & member_sets["test"]),
            "train_test_fraction_of_test_unique": (
                len(member_sets["train"] & member_sets["test"]) /
                len(member_sets["test"]) if member_sets["test"] else None),
        },
        "shared_source_nodes_train_test": len(node_sets["train"] & node_sets["test"]),
    }


def run_real_p1(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run shared P1 variants on fixed real temporal windows from two sources."""
    config_path, output_dir = Path(config_path).resolve(), Path(output_dir).resolve()
    config = load_real_protocol(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {output_dir}")
    profiler = RunProfiler()
    windows: list[GraphRecord] = []
    audits: list[dict[str, Any]] = []
    windows_by_source: dict[str, list[GraphRecord]] = {}
    for dataset in config["datasets"]:
        files = _resolve_local_files(config_path, dataset)
        dataset["files"] = {key: str(path) for key, path in files.items()}
        archive_manifest = _archive_audit(dataset, files)
        with profiler.stage("load_and_split:" + dataset["source_id"]):
            source = load_cornell_event_files(
                nverts_path=str(files["nverts"]),
                simplices_path=str(files["simplices"]),
                times_path=str(files["times"]),
                source_id=dataset["source_id"],
                source_url=dataset["source_url"],
                license_name=dataset.get("license_or_access_terms"),
                min_event_size=2,
            )
            if (dataset.get("expected_raw_event_count") is not None and
                    source.provenance["raw_event_count"] != dataset["expected_raw_event_count"]):
                raise ValueError(f"{dataset['source_id']}: raw event count does not match official release")
            if (dataset.get("expected_num_nodes") is not None and
                    source.num_nodes != dataset["expected_num_nodes"]):
                raise ValueError(f"{dataset['source_id']}: node count does not match official release")
            if dataset.get("windowing") is not None:
                dataset["window_boundaries"] = _quantile_time_boundaries(
                    source, dataset["windowing"]["window_count"])
                dataset["windowing_rule_used_to_lock_boundaries"] = dataset.pop("windowing")
            source_windows = split_events_into_windows(
                source, boundaries=dataset["window_boundaries"],
                labels=dataset["window_labels"])
        windows.extend(source_windows)
        windows_by_source[dataset["source_id"]] = source_windows
        audits.append(_source_audit(source, source_windows, archive_manifest))
    validate_disjoint_graph_splits({role: [graph.graph_id for graph in windows
                                           if graph.split == role]
                                    for role in ("train", "validation", "test")})
    validate_event_disjointness({role: [event_id for graph in windows
                                          if graph.split == role
                                          for event_id in graph.hyperedge_event_ids or []]
                                 for role in ("train", "validation", "test")})
    if sum(len(graph.hyperedges_truth or []) for graph in windows) != sum(
            item["retained_event_count"] for item in audits):
        raise AssertionError("the time windows must preserve all retained events")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(output_dir / "environment.json", environment_record())
    write_json(output_dir / "real_input_audit.json", {
        "protocol": config["project"]["protocol"],
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "minimum_meaningful_delta_ap_fixed_before_scoring": config["minimum_meaningful_delta_ap"],
        "processing_code_sha256": {
            "real_pipeline": sha256_file(Path(__file__)),
            "real_data": sha256_file(Path(__file__).with_name("real_data.py")),
        },
        "event_handling": "complete events, unique member sets for binary target",
        "primary_analysis_unit": "source_dataset",
        "datasets": audits,
    })
    source_reports: dict[str, Any] = {}
    comparisons = ("Q3_vs_C0_classical", "Q3_vs_C2_wave", "Q3_vs_C3_spectral")
    for index, dataset in enumerate(config["datasets"]):
        source_id = dataset["source_id"]
        source_output = output_dir / "sources" / f"source-{index}"
        source_output.mkdir(parents=True, exist_ok=True)
        source_config = dict(config)
        source_config["datasets"] = [dataset]
        (source_output / "config_resolved.yaml").write_text(
            yaml.safe_dump(source_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8")
        source_profiler = RunProfiler()
        source_report = run_p1_graphs(
            config_path, source_output, source_config, windows_by_source[source_id],
            source_profiler,
            gate_reason="One temporal source cannot independently establish Gate 1; "
                        "evaluate both sources, source-level replication and cost",
            inference_note="Time windows from this source are correlated. All test "
                           "windows share one source cluster, so a source-level "
                           "confidence interval is undefined.",
            comparison_group_by_graph={graph.graph_id: source_id
                                       for graph in windows_by_source[source_id]},
        )
        source_report["minimum_meaningful_delta_ap"] = config["minimum_meaningful_delta_ap"]
        write_json(source_output / "candidate_metrics.json", source_report)
        source_reports[source_id] = {
            "output": str(source_output),
            "test_window_count": sum(graph.split == "test" for graph in windows_by_source[source_id]),
            "Q3_test_AP": source_report["variants"]["Q3_group"]
                          ["test_candidate_metrics_pooled"]["ap"],
            "test_positive_rate": source_report["variants"]["Q3_group"]
                                  ["test_candidate_metrics_pooled"]["positive_rate"],
            "planned_delta_ap": {name: source_report["planned_comparisons"][name]["delta"]
                                 for name in comparisons},
            "delta_reaches_fixed_minimum": {
                name: (None if source_report["planned_comparisons"][name]["delta"] is None
                       else source_report["planned_comparisons"][name]["delta"] >=
                            config["minimum_meaningful_delta_ap"])
                for name in comparisons},
            "planned_comparison_details": source_report["planned_comparisons"],
        }
    equal_source_mean = {}
    for name in comparisons:
        deltas = [entry["planned_delta_ap"][name]
                  for entry in source_reports.values()]
        equal_source_mean[name] = (float(np.mean(deltas))
                                   if all(value is not None for value in deltas)
                                   else None)
    all_sources_reach_minimum = {
        name: (None if any(entry["delta_reaches_fixed_minimum"][name] is None
                           for entry in source_reports.values())
               else all(entry["delta_reaches_fixed_minimum"][name]
                        for entry in source_reports.values()))
        for name in comparisons
    }
    report = {
        "protocol": config["project"]["protocol"],
        "primary_metric": "average_precision",
        "primary_analysis_unit": "source_dataset",
        "minimum_meaningful_delta_ap": config["minimum_meaningful_delta_ap"],
        "sources": source_reports,
        "equal_source_mean_delta_ap": equal_source_mean,
        "all_sources_reach_fixed_minimum": all_sources_reach_minimum,
        "gate_1": {
            "status": "not_evaluated",
            "reason": "Two temporal pilot datasets and their correlated time windows do not "
                      "provide enough independent replication or cost evidence to "
                      "declare a robust Gate 1 result.",
        },
    }
    write_json(output_dir / "candidate_metrics.json", report)
    write_json(output_dir / "data_manifest.json", {
        "protocol": config["project"]["protocol"],
        "config_sha256": sha256_file(config_path),
        "resolved_config_sha256": sha256_file(output_dir / "config_resolved.yaml"),
        "source_ids": list(source_reports),
        "source_output_paths": {source_id: entry["output"]
                                for source_id, entry in source_reports.items()},
        "archive_and_window_audit": "real_input_audit.json",
    })
    write_json(output_dir / "runtime_preparation.json", profiler.record())
    return report
