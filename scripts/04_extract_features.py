#!/usr/bin/env python3
"""Extract label-free C0/C1/C2/C3/Q1/Q3 features for a fixed candidate file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import sparse

from qhyper.classic_features import classical_candidate_matrix, dynamic_family_matrix
from qhyper.experiment_logger import RunProfiler, sha256_file, write_json
from qhyper.group_response import extract_group_quantum_features
from qhyper.hamiltonian import build_hamiltonian


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-npz", required=True, type=Path)
    parser.add_argument("--candidates-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="new .npz feature cache path")
    parser.add_argument("--times", nargs="+", type=float,
                        default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--max-state-columns", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output))
    A = sparse.load_npz(args.observed_npz)
    payload = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    if sha256_file(args.observed_npz) != payload.get("observed_sha256"):
        raise ValueError("candidate file was generated from a different observed graph")
    rows = payload["candidates"]
    candidates = [tuple(row["node_ids"]) for row in rows]
    sources = [tuple(row["proposal_sources"]) for row in rows]
    if not candidates:
        raise ValueError("empty candidate file")
    profiler = RunProfiler()
    H = build_hamiltonian(A)
    with profiler.stage("C0_classical"):
        C0 = classical_candidate_matrix(A, candidates, sources=sources)
    with profiler.stage("Q3_group"):
        Q3, diagnostics = extract_group_quantum_features(
            H, candidates, args.times, exact_incoherent=True,
            max_state_columns=args.max_state_columns)
    controls = {}
    for family in ("heat", "wave", "spectral", "quantum_pair"):
        with profiler.stage(family):
            controls[family] = dynamic_family_matrix(
                A, H, candidates, args.times, family,
                max_state_columns=args.max_state_columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, C0_classical=C0, Q3_group=Q3,
                        C1_heat=controls["heat"], C2_wave=controls["wave"],
                        C3_spectral=controls["spectral"],
                        Q1_pairwise=controls["quantum_pair"],
                        R_coh=diagnostics["R_coh"], R_inc=diagnostics["R_inc"],
                        coherence_contrast=diagnostics["C"])
    write_json(args.output.with_suffix(".manifest.json"), {
        "observed_sha256": sha256_file(args.observed_npz),
        "candidates_sha256": sha256_file(args.candidates_json),
        "candidate_count": len(candidates),
        "time_grid": args.times,
        "contains_truth_or_labels": False,
        "runtime_memory": profiler.record(),
    })
    print("Extracted features for {} candidates".format(len(candidates)))


if __name__ == "__main__":
    main()
