import importlib.util
import io
from pathlib import Path
import tarfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "06_unpack_cornell_events.py"
SPEC = importlib.util.spec_from_file_location("unpack_cornell_events", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _archive(path, entries):
    with tarfile.open(path, "w:gz") as tar:
        for name, content in entries:
            encoded = content.encode("utf-8")
            entry = tarfile.TarInfo(name)
            entry.size = len(encoded)
            tar.addfile(entry, io.BytesIO(encoded))


def test_unpack_uses_canonical_paths_and_records_hashes(tmp_path):
    archive = tmp_path / "source.tar.gz"
    _archive(archive, [("../../source-nverts.txt", "3\n"),
                       ("data/source-simplices.txt", "1 2 3\n"),
                       ("data/source-times.txt", "1\n")])
    output = tmp_path / "raw"
    manifest = MODULE.unpack(
        archive, output, source_id="test/source/temporal",
        source_url="https://example.org/source",
        archive_url="https://example.org/archive")
    assert (output / "nverts.txt").read_text() == "3\n"
    assert not (tmp_path.parent / "source-nverts.txt").exists()
    assert manifest["archive_sha256"] == MODULE._hash_file(archive)
    assert manifest["files"]["simplices"]["sha256"] == MODULE._hash_file(
        output / "simplices.txt")


def test_ambiguous_archive_is_rejected_without_output(tmp_path):
    archive = tmp_path / "source.tar.gz"
    _archive(archive, [("a-nverts.txt", "2"), ("b-nverts.txt", "2"),
                       ("x-simplices.txt", "1 2"), ("x-times.txt", "1")])
    output = tmp_path / "raw"
    with pytest.raises(ValueError, match="exactly one nverts"):
        MODULE.unpack(archive, output, source_id="test/source/temporal",
                      source_url="https://example.org/source",
                      archive_url="https://example.org/archive")
    assert not output.exists()
