"""Prespecified random-seed stability check for the temporal real P1 pilot.

Seeds change candidate proposal and stochastic reference calculations, not the
underlying time windows. Their variation is reported as a sensitivity check;
seeds are not independent datasets or valid bootstrap units.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .experiment_logger import sha256_file, write_json
from .real_pipeline import load_real_protocol, run_real_p1


def run_real_seed_sweep(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run the fixed seed list and summarize per-source paired AP deltas."""
    config_path, output_dir = Path(config_path).resolve(), Path(output_dir).resolve()
    base = load_real_protocol(config_path)
    seeds = base.get("replicate_seeds")
    if (not isinstance(seeds, list) or len(seeds) < 3 or
            any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ValueError("replicate_seeds must contain at least three distinct integers")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty sweep directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "seed_configs"
    config_dir.mkdir()
    runs: dict[int, dict[str, Any]] = {}
    locked_boundaries = None
    for seed in seeds:
        config = deepcopy(base)
        config["project"]["seed"] = seed
        for dataset in config["datasets"]:
            dataset["files"] = {
                key: str((config_path.parent / value).resolve())
                for key, value in dataset["files"].items()
            }
        seed_config = config_dir / f"seed-{seed}.yaml"
        seed_config.write_text(yaml.safe_dump(config, sort_keys=False,
                                              allow_unicode=True), encoding="utf-8")
        run_output = output_dir / "runs" / f"seed-{seed}"
        result = run_real_p1(seed_config, run_output)
        locked = yaml.safe_load((run_output / "config_resolved.yaml").read_text(
            encoding="utf-8"))
        boundaries = tuple(tuple(item["window_boundaries"])
                           for item in locked["datasets"])
        if locked_boundaries is None:
            locked_boundaries = boundaries
        elif boundaries != locked_boundaries:
            raise AssertionError("source time boundaries changed across seed runs")
        runs[seed] = result

    comparisons = ("Q3_vs_C0_classical", "Q3_vs_C2_wave", "Q3_vs_C3_spectral")
    sources = {}
    for source_id in runs[seeds[0]]["sources"]:
        comparisons_by_source = {}
        for name in comparisons:
            values = [runs[seed]["sources"][source_id]["planned_delta_ap"][name]
                      for seed in seeds]
            valid = [float(value) for value in values if value is not None]
            comparisons_by_source[name] = {
                "delta_ap_by_seed": {str(seed): value for seed, value in zip(seeds, values)},
                "mean": float(np.mean(valid)) if len(valid) == len(seeds) else None,
                "sample_sd": (float(np.std(valid, ddof=1))
                              if len(valid) == len(seeds) else None),
                "minimum": min(valid) if len(valid) == len(seeds) else None,
                "maximum": max(valid) if len(valid) == len(seeds) else None,
            }
        sources[source_id] = {
            "Q3_test_AP_by_seed": {
                str(seed): runs[seed]["sources"][source_id]["Q3_test_AP"]
                for seed in seeds
            },
            "planned_comparisons": comparisons_by_source,
        }
    summary = {
        "protocol": base["project"]["protocol"],
        "seed_list": seeds,
        "config_sha256": sha256_file(config_path),
        "locked_window_boundaries": locked_boundaries,
        "minimum_meaningful_delta_ap": base["minimum_meaningful_delta_ap"],
        "sources": sources,
        "inference_note": "Seed variability is a stability diagnostic on the same "
                          "two source datasets, not independent replication.",
        "gate_1": {"status": "not_evaluated",
                   "reason": "The real-data results and computation cost require review; "
                             "repeated seeds are not new source datasets."},
    }
    write_json(output_dir / "seed_sweep_summary.json", summary)
    return summary
