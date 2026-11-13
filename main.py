"""Small PyCharm-friendly entry point for the staged research workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from qhyper.p1_pipeline import run_p1
from qhyper.real_pipeline import run_real_p1
from qhyper.real_sweep import run_real_seed_sweep


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum-inspired hypergraph P0/P1 research")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("verify", help="run the independent P0 mathematical checks")
    p1 = subcommands.add_parser("p1", help="run cross-graph synthetic P1 discrimination")
    p1.add_argument("--config", type=Path, default=ROOT / "configs" / "p1_smoke.yaml")
    p1.add_argument("--output", type=Path)
    p1_real = subcommands.add_parser(
        "p1-real", help="run temporal real-data P1 discrimination")
    p1_real.add_argument("--config", type=Path,
                         default=ROOT / "configs" / "p1_real_pilot.yaml")
    p1_real.add_argument("--output", type=Path)
    p1_sweep = subcommands.add_parser(
        "p1-real-sweep", help="run prespecified real-data P1 seed stability checks")
    p1_sweep.add_argument("--config", type=Path,
                          default=ROOT / "configs" / "p1_real_pilot.yaml")
    p1_sweep.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        subprocess.run([sys.executable, str(ROOT / "scripts" / "00_verify_theory.py")],
                       check=True)
        return
    prefix = ("p1-real-sweep" if args.command == "p1-real-sweep" else
              "p1-real" if args.command == "p1-real" else "p1")
    output = args.output or ROOT / "results" / datetime.now(timezone.utc).strftime(
        prefix + "-%Y%m%dT%H%M%SZ")
    if args.command == "p1-real-sweep":
        sweep = run_real_seed_sweep(args.config, output)
        print(json.dumps({"output": str(output), "gate_1": sweep["gate_1"],
                          "seeds": sweep["seed_list"],
                          "sources": list(sweep["sources"])},
                         ensure_ascii=False, indent=2))
        return
    report = (run_real_p1(args.config, output) if args.command == "p1-real"
              else run_p1(args.config, output))
    summary = {"output": str(output), "gate_1": report["gate_1"]}
    if args.command == "p1-real":
        summary["Q3_test_AP_by_source"] = {
            source: values["Q3_test_AP"]
            for source, values in report["sources"].items()}
        summary["equal_source_mean_delta_ap"] = report["equal_source_mean_delta_ap"]
    else:
        summary["Q3_test_AP"] = (report["variants"]["Q3_group"]
                                 ["test_candidate_metrics_pooled"]["ap"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
