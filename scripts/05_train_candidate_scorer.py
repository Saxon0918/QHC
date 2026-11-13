#!/usr/bin/env python3
"""Run the fixed P1 synthetic candidate-discrimination protocol."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qhyper.p1_pipeline import run_p1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "p1_smoke.yaml")
    parser.add_argument("--output", type=Path, default=None,
                        help="New run directory; existing nonempty directories are rejected")
    arguments = parser.parse_args()
    output = arguments.output
    if output is None:
        run_id = datetime.now(timezone.utc).strftime("p1-%Y%m%dT%H%M%SZ")
        output = ROOT / "results" / run_id
    report = run_p1(arguments.config, output)
    concise = {
        "output": str(output),
        "protocol": report["protocol"],
        "gate_1": report["gate_1"],
        "test_ap": {name: values["test_candidate_metrics_pooled"]["ap"]
                    for name, values in report["variants"].items()},
        "planned_comparisons": report["planned_comparisons"],
    }
    print(json.dumps(concise, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
