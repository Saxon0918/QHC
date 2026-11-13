#!/usr/bin/env python3
"""Safely unpack the three event files from a verified Cornell tar.gz archive.

Download the archive from its dataset's official release page yourself. This
script reads named regular members only; it never extracts archive paths.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Optional


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack(archive: Path, output: Path, *, source_id: str,
           source_url: str, archive_url: str,
           downloaded_at_utc: Optional[str] = None) -> dict:
    archive, output = Path(archive), Path(output)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    suffixes = {"nverts": "nverts.txt", "simplices": "simplices.txt",
                "times": "times.txt"}
    output.parent.mkdir(parents=True, exist_ok=True)
    extracted = {}
    with tarfile.open(archive, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        selected = {}
        for key, suffix in suffixes.items():
            matches = [member for member in members
                       if Path(member.name).name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"expected exactly one {suffix} regular file; found {len(matches)}")
            selected[key] = matches[0]
        with tempfile.TemporaryDirectory(prefix="cornell-events-", dir=output.parent) as temporary:
            staging = Path(temporary)
            for key, member in selected.items():
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read archive member {member.name}")
                destination = staging / (key + ".txt")
                with source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                extracted[key] = {
                    "archive_member": member.name,
                    "file": destination.name,
                    "size_bytes": destination.stat().st_size,
                    "sha256": _hash_file(destination),
                }
            manifest = {
                "source_id": source_id,
                "source_url": source_url,
                "archive_url": archive_url,
                "archive_file_name": archive.name,
                "archive_sha256": _hash_file(archive),
                "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
                "downloaded_at_utc": downloaded_at_utc,
                "files": extracted,
            }
            (staging / "archive_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8")
            staging.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--downloaded-at-utc", default=None,
                        help="Actual ISO-8601 archive download time, if known")
    args = parser.parse_args()
    result = unpack(args.archive, args.output, source_id=args.source_id,
                    source_url=args.source_url, archive_url=args.archive_url,
                    downloaded_at_utc=args.downloaded_at_utc)
    print(json.dumps({"output": str(args.output), "archive_sha256": result["archive_sha256"],
                      "files": {key: item["sha256"] for key, item in result["files"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
