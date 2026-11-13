"""Small PyCharm-friendly entry point for the staged research workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from qhyper.p1_pipeline import run_p1


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum-inspired hypergraph P0/P1 research")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("verify", help="run the independent P0 mathematical checks")
    p1 = subcommands.add_parser("p1", help="run cross-graph synthetic P1 discrimination")
    p1.add_argument("--config", type=Path, default=ROOT / "configs" / "p1_smoke.yaml")
    p1.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        subprocess.run([sys.executable, str(ROOT / "scripts" / "00_verify_theory.py")],
                       check=True)
        return
    output = args.output or ROOT / "results" / datetime.now(timezone.utc).strftime(
        "p1-%Y%m%dT%H%M%SZ")
    report = run_p1(args.config, output)
    print(json.dumps({"output": str(output), "gate_1": report["gate_1"],
                      "Q3_test_AP": report["variants"]["Q3_group"]
                      ["test_candidate_metrics_pooled"]["ap"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
