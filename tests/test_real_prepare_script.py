"""End-to-end fixture checks for the offline Cornell data preparation CLI."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from scipy import sparse


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "02_prepare_real_data.py"


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    nverts = tmp_path / "toy-nverts.txt"
    simplices = tmp_path / "toy-simplices.txt"
    times = tmp_path / "toy-times.txt"
    nverts.write_text("3\n1\n3\n2\n", encoding="utf-8")
    simplices.write_text("10\n20\n30\n99\n10\n20\n30\n20\n40\n", encoding="utf-8")
    times.write_text("100\n101\n102\n103\n", encoding="utf-8")
    return nverts, simplices, times


def _command(nverts: Path, simplices: Path, times: Path, output: Path) -> list[str]:
    return [sys.executable, str(SCRIPT),
            "--nverts", str(nverts), "--simplices", str(simplices),
            "--times", str(times),
            "--source-id", "cornell/toy/temporal-v1",
            "--source-url", "https://example.test/cornell/toy/temporal-v1",
            "--license", "test-only", "--output", str(output)]


def test_offline_cli_separates_observation_truth_and_manifest(tmp_path):
    nverts, simplices, times = _fixture(tmp_path)
    output = tmp_path / "prepared"
    result = subprocess.run(_command(nverts, simplices, times, output),
                            capture_output=True, text=True, check=True)
    summary = json.loads(result.stdout)
    assert summary["retained_event_count"] == 3
    assert summary["num_nodes"] == 5
    observed_file = output / "observed" / "adjacency_binary.npz"
    observed_meta = json.loads((output / "observed" / "graph_metadata.json").read_text())
    truth_file = output / "truth" / "hyperedges.json"
    truth = json.loads(truth_file.read_text())
    manifest = json.loads((output / "data_manifest.json").read_text())
    A = sparse.load_npz(observed_file)
    assert A.shape == (5, 5)
    assert A[0, 1] == A[0, 2] == A[1, 2] == A[1, 3] == 1
    assert A[4, :].nnz == 0  # singleton-only source node remains isolated
    assert observed_meta["original_node_ids"] == [10, 20, 30, 40, 99]
    assert observed_meta["split"] == "unsplit"
    assert "events" not in observed_meta
    assert truth["event_count"] == 3
    assert truth["unique_member_set_count"] == 2
    assert [event["node_ids"] for event in truth["events"]] == [
        [0, 1, 2], [0, 1, 2], [1, 3]]
    assert [event["timestamp"] for event in truth["events"]] == [100, 102, 103]
    assert manifest["source_raw_event_count"] == 4
    assert manifest["excluded_singleton_event_count"] == 1
    assert manifest["processing"]["dataset_split"] == "unsplit"
    assert manifest["source_files"]["simplices"]["sha256"] == sha256(simplices.read_bytes()).hexdigest()
    assert manifest["outputs"]["truth_events"]["sha256"] == sha256(truth_file.read_bytes()).hexdigest()


def test_cli_rejects_nonempty_output_without_overwriting(tmp_path):
    nverts, simplices, times = _fixture(tmp_path)
    output = tmp_path / "prepared"
    command = _command(nverts, simplices, times, output)
    subprocess.run(command, capture_output=True, text=True, check=True)
    manifest_path = output / "data_manifest.json"
    original = manifest_path.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "refusing to overwrite nonempty output" in repeated.stderr
    assert manifest_path.read_bytes() == original


def test_cli_rejects_unqualified_data_identity(tmp_path):
    nverts, simplices, times = _fixture(tmp_path)
    output = tmp_path / "invalid"
    command = _command(nverts, simplices, times, output)
    command[command.index("--source-id") + 1] = "email-Enron"
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "source_id must be version-qualified" in rejected.stderr
    assert not output.exists()
