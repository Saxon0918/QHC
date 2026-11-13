#!/usr/bin/env python3
"""Materialize independent S1/S2 graphs and the S3 ambiguity witness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import sparse

from qhyper.experiment_logger import sha256_file, write_json
from qhyper.p1_pipeline import load_protocol, make_synthetic_graphs
from qhyper.synthetic import triangle_ambiguity_example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_protocol(args.config)
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty output: {}".format(output))
    (output / "observed").mkdir(parents=True)
    (output / "truth_for_supervision_and_evaluation").mkdir()
    manifest = {"config_sha256": sha256_file(args.config), "graphs": []}
    for graph in make_synthetic_graphs(config):
        observed = graph.observed()
        path = output / "observed" / (graph.graph_id + ".npz")
        sparse.save_npz(path, observed.adjacency_csr)
        if observed.features is not None:
            np.save(output / "observed" / (graph.graph_id + "_features.npy"),
                    observed.features)
        truth_path = output / "truth_for_supervision_and_evaluation" / (
            graph.graph_id + ".json")
        write_json(truth_path, {"graph_id": graph.graph_id, "split": graph.split,
                                "hyperedges_truth": graph.hyperedges_truth,
                                "hyperedge_event_ids": graph.hyperedge_event_ids})
        manifest["graphs"].append({
            "graph_id": graph.graph_id, "split": graph.split,
            "provenance": graph.provenance,
            "observed_adjacency": str(path.relative_to(output)),
            "observed_sha256": sha256_file(path),
            "truth_file": str(truth_path.relative_to(output)),
            "truth_sha256": sha256_file(truth_path),
            "truth_is_not_a_feature_input": True,
        })
    first, second = triangle_ambiguity_example()
    sparse.save_npz(output / "observed" / "S3_shared_projection.npz",
                    first.adjacency_csr)
    write_json(output / "truth_for_supervision_and_evaluation" /
               "S3_two_worlds.json", {
                   "world_one": first.hyperedges_truth,
                   "world_two": second.hyperedges_truth,
                   "same_binary_projection": True,
                   "interpretation": "不能仅从 A 区分两个真实超图",
               })
    write_json(output / "data_manifest.json", manifest)
    print("Generated {} independent graphs in {}".format(len(manifest["graphs"]), output))


if __name__ == "__main__":
    main()
