"""Export PyG Planetoid public-split graphs to the B1 NPZ contract.

Requires a separately installed, version-compatible PyTorch Geometric.  This
script performs the dataset download only when the user runs it.  It preserves
the dataset's public masks and does not create or inspect a hyperedge truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

# Direct execution adds scripts/ (not the repository root) to sys.path.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qhyper.experiment_logger import sha256_file, write_json
from qhyper.hamiltonian import checked_adjacency


def save_b1_npz(
    output: Path,
    edge_index: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, int]:
    """Validate and export one undirected binary graph with explicit masks."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or x.shape[0] == 0 or not np.isfinite(x).all():
        raise ValueError("features must be a finite N-by-D matrix")
    n = x.shape[0]
    if y.shape != (n,):
        raise ValueError("labels must have one entry per node")
    endpoints = np.asarray(edge_index)
    if (endpoints.ndim != 2 or endpoints.shape[0] != 2
            or not np.issubdtype(endpoints.dtype, np.integer)
            or np.any(endpoints < 0) or np.any(endpoints >= n)):
        raise ValueError("edge_index must be a 2-by-E integer node-index array")
    graph = sparse.csr_matrix((np.ones(endpoints.shape[1], dtype=np.float64),
                               (endpoints[0], endpoints[1])), shape=(n, n))
    graph = sparse.triu(graph.maximum(graph.T), k=1, format="csr")
    graph = (graph + graph.T).tocsr()
    graph.data[:] = 1.0
    graph = checked_adjacency(graph)
    roles = {}
    for name, mask in (("train", train_mask), ("validation", validation_mask),
                       ("test", test_mask)):
        values = np.asarray(mask)
        if values.shape != (n,) or values.dtype != np.bool_:
            raise ValueError(f"{name}_mask must be a length-N Boolean vector")
        indices = np.flatnonzero(values).astype(np.int64)
        if indices.size == 0:
            raise ValueError(f"{name}_mask is empty")
        roles[name] = indices
    all_indices = np.concatenate(list(roles.values()))
    if np.unique(all_indices).size != all_indices.size:
        raise ValueError("public train/validation/test masks overlap")
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("output must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        adjacency_data=graph.data,
        adjacency_indices=graph.indices,
        adjacency_indptr=graph.indptr,
        adjacency_shape=np.asarray(graph.shape, dtype=np.int64),
        features=x,
        labels=y,
        train_idx=roles["train"],
        validation_idx=roles["validation"],
        test_idx=roles["test"],
    )
    return {
        "n_nodes": n,
        "n_edges": graph.nnz // 2,
        "n_features": x.shape[1],
        "n_train": len(roles["train"]),
        "n_validation": len(roles["validation"]),
        "n_test": len(roles["test"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("Cora", "CiteSeer", "PubMed"),
                        default="Cora")
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "raw" / "planetoid")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        import torch_geometric
        from torch_geometric.datasets import Planetoid
    except ImportError as exc:
        raise SystemExit("Install a PyTorch-compatible torch-geometric on the server first") from exc
    dataset = Planetoid(root=str(args.root), name=args.dataset, split="public")
    data = dataset[0]
    output = args.output or ROOT / "data" / "processed" / f"{args.dataset.lower()}_b1.npz"
    counts = save_b1_npz(
        output,
        data.edge_index.detach().cpu().numpy(),
        data.x.detach().cpu().numpy(),
        data.y.detach().cpu().numpy(),
        data.train_mask.detach().cpu().numpy(),
        data.val_mask.detach().cpu().numpy(),
        data.test_mask.detach().cpu().numpy(),
    )
    raw_dir = Path(dataset.raw_dir)
    manifest = {
        "dataset": args.dataset,
        "source": "torch_geometric.datasets.Planetoid",
        "source_url": "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html",
        "split": "public",
        "torch_geometric_version": torch_geometric.__version__,
        "output": str(Path(output).resolve()),
        "output_sha256": sha256_file(output),
        "raw_file_sha256": {str(path.name): sha256_file(path)
                            for path in sorted(raw_dir.iterdir()) if path.is_file()},
        **counts,
    }
    write_json(Path(output).with_suffix(".manifest.json"), manifest)
    print(f"Wrote {output} ({counts['n_nodes']} nodes, {counts['n_edges']} edges)")


if __name__ == "__main__":
    main()
