#!/usr/bin/env python3
"""Prepare local Cornell event files as separate binary observations and truth.

This command never downloads data and never splits or truncates an event. The
caller must name the exact data source/version and supply local source files.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qhyper.graph_io import load_cornell_event_files


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True,
                               ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def prepare_real_data(*, nverts: Path, simplices: Path,
                      source_id: str, source_url: str, output: Path,
                      times: Optional[Path] = None,
                      node_universe: Optional[Path] = None,
                      license_name: Optional[str] = None) -> dict[str, Any]:
    """Write one complete event dataset with hidden truth in its own folder.

    The output is deliberately unsplit. Caller-specified ``source_id`` must be
    version-qualified (for example ``cornell/email-Enron/temporal``) so a SNAP
    ordinary graph with the same basename cannot be silently merged.
    """
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    original_ids: Optional[Sequence[int]] = None
    if node_universe is not None:
        try:
            original_ids = [int(token) for token in Path(node_universe).read_text(
                encoding="utf-8").split()]
        except ValueError as exc:
            raise ValueError("node universe must contain integer source IDs") from exc
        if not original_ids:
            raise ValueError("node universe cannot be empty")
    record = load_cornell_event_files(
        nverts_path=str(nverts), simplices_path=str(simplices),
        source_id=source_id, source_url=source_url,
        times_path=None if times is None else str(times),
        original_node_ids=original_ids, license_name=license_name,
        min_event_size=2,
    )
    observed = record.observed()
    output.mkdir(parents=True, exist_ok=True)
    observed_dir = output / "observed"
    truth_dir = output / "truth"
    observed_dir.mkdir(exist_ok=True)
    truth_dir.mkdir(exist_ok=True)
    adjacency_path = observed_dir / "adjacency_binary.npz"
    metadata_path = observed_dir / "graph_metadata.json"
    truth_path = truth_dir / "hyperedges.json"
    sparse.save_npz(adjacency_path, observed.adjacency_csr)
    _write_json(metadata_path, {
        "graph_id": observed.graph_id,
        "num_nodes": observed.num_nodes,
        "observation_mode": "binary",
        "original_node_ids": list(observed.original_node_ids),
        "split": "unsplit",
        "source_id": source_id,
        "source_url": source_url,
    })
    events = [
        {"event_id": record.hyperedge_event_ids[i],
         "node_ids": list(edge),
         "timestamp": None if record.hyperedge_timestamps is None
         else record.hyperedge_timestamps[i]}
        for i, edge in enumerate(record.hyperedges_truth or [])
    ]
    _write_json(truth_path, {
        "graph_id": record.graph_id,
        "node_id_space": "internal_zero_based",
        "event_count": len(events),
        "unique_member_set_count": len({tuple(event["node_ids"]) for event in events}),
        "events": events,
    })
    manifest = {
        "source_id": source_id,
        "source_url": source_url,
        "license": license_name,
        "source_files": record.provenance["files"],
        "source_raw_event_count": record.provenance["raw_event_count"],
        "retained_event_count": len(events),
        "excluded_singleton_event_count": record.provenance["excluded_small_event_count"],
        "processing": {
            "projection": "binary_clique",
            "minimum_retained_event_size": 2,
            "event_handling": "complete source events; no event split or node intersection",
            "repeated_member_sets": "preserved as separate events in truth",
            "dataset_split": "unsplit",
        },
        "num_nodes": observed.num_nodes,
        "outputs": {
            "observed_adjacency": {"path": str(adjacency_path.relative_to(output)),
                                   "sha256": _sha256_file(adjacency_path)},
            "observed_metadata": {"path": str(metadata_path.relative_to(output)),
                                  "sha256": _sha256_file(metadata_path)},
            "truth_events": {"path": str(truth_path.relative_to(output)),
                             "sha256": _sha256_file(truth_path)},
        },
    }
    if node_universe is not None:
        manifest["source_files"]["node_universe"] = {
            "name": Path(node_universe).name,
            "sha256": _sha256_file(Path(node_universe)),
        }
    _write_json(output / "data_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nverts", type=Path, required=True,
                        help="Local Cornell event-size file (*-nverts.txt)")
    parser.add_argument("--simplices", type=Path, required=True,
                        help="Local flattened event-member file (*-simplices.txt)")
    parser.add_argument("--times", type=Path, default=None,
                        help="Optional local event-time file (*-times.txt)")
    parser.add_argument("--node-universe", type=Path, default=None,
                        help="Optional local list of all original node IDs, including isolated nodes")
    parser.add_argument("--source-id", required=True,
                        help="Version-qualified ID, e.g. cornell/email-Enron/temporal")
    parser.add_argument("--source-url", required=True,
                        help="Exact data release page URL for provenance")
    parser.add_argument("--license", dest="license_name", default=None,
                        help="Dataset license or access terms, if known")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; nonempty directories are rejected")
    args = parser.parse_args()
    try:
        manifest = prepare_real_data(
            nverts=args.nverts, simplices=args.simplices,
            times=args.times, node_universe=args.node_universe,
            source_id=args.source_id, source_url=args.source_url,
            license_name=args.license_name, output=args.output,
        )
    except (ValueError, FileExistsError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"output": str(args.output),
                      "source_id": manifest["source_id"],
                      "num_nodes": manifest["num_nodes"],
                      "retained_event_count": manifest["retained_event_count"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
