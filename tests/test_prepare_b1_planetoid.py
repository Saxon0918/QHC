"""The optional real-data exporter must honor the B1 loader contract."""

import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path

from qhyper.construction_pipeline import _load_npz_graph
from scripts.prepare_b1_planetoid import save_b1_npz


def test_direct_script_invocation_finds_qhyper_without_editable_install(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_b1_planetoid.py"
    completed = subprocess.run(
        [sys.executable, "-I", str(script), "--help"],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert "--dataset" in completed.stdout


def test_planetoid_export_round_trips_binary_graph_and_masks(tmp_path):
    edges = np.array([[0, 1, 2, 3, 4, 5, 0],
                      [1, 2, 3, 4, 5, 0, 0]], dtype=np.int64)
    x = np.arange(18, dtype=float).reshape(6, 3)
    y = np.array([0, 1, 0, 1, 0, 1])
    masks = [np.array([1, 1, 0, 0, 0, 0], dtype=bool),
             np.array([0, 0, 1, 1, 0, 0], dtype=bool),
             np.array([0, 0, 0, 0, 1, 1], dtype=bool)]
    output = tmp_path / "sample.npz"
    counts = save_b1_npz(output, edges, x, y, *masks)
    graph, loaded_x, loaded_y, split, metadata = _load_npz_graph(output)
    assert counts["n_edges"] == 6
    assert graph.nnz == 12 and graph.diagonal().sum() == 0
    assert np.array_equal(loaded_x, x) and np.array_equal(loaded_y, y)
    assert split["train"].tolist() == [0, 1]
    assert split["test"].tolist() == [4, 5]
    assert metadata["source_sha256"]


def test_export_rejects_overlapping_masks(tmp_path):
    edges = np.array([[0], [1]], dtype=np.int64)
    x = np.ones((6, 2))
    y = np.array([0, 1, 0, 1, 0, 1])
    train = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
    validation = np.array([0, 1, 1, 0, 0, 0], dtype=bool)
    test = np.array([0, 0, 0, 0, 1, 1], dtype=bool)
    with pytest.raises(ValueError, match="overlap"):
        save_b1_npz(tmp_path / "sample.npz", edges, x, y,
                    train, validation, test)
