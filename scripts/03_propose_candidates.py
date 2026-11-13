#!/usr/bin/env python3
"""Generate a truth-free, fixed-budget clique candidate pool from one graph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy import sparse

from qhyper.candidate_proposal import propose_candidates
from qhyper.experiment_logger import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-npz", required=True, type=Path)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generators", nargs="+", default=["local_clique", "maximal_clique_subsets"],
                        choices=["local_clique", "maximal_clique_subsets"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output))
    A = sparse.load_npz(args.observed_npz)
    candidates = propose_candidates(
        A, budget=args.budget, min_k=args.min_size, max_k=args.max_size,
        seed=args.seed, generators=args.generators,
        graph_id=args.graph_id, return_records=True)
    write_json(args.output, {
        "graph_id": args.graph_id,
        "observed_sha256": sha256_file(args.observed_npz),
        "seed": args.seed,
        "requested_budget": args.budget,
        "actual_candidates": len(candidates),
        "min_size": args.min_size,
        "max_size": args.max_size,
        "generators": args.generators,
        "candidates": [record.feature_payload() for record in candidates],
    })
    print("Proposed {} candidates".format(len(candidates)))


if __name__ == "__main__":
    main()
