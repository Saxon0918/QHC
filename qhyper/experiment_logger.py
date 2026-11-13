"""Small, explicit provenance and resource records for reproducible runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import psutil
import scipy
import sklearn


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Not JSON serializable: {}".format(type(value).__name__))


def write_json(path: Path, payload: Any) -> None:
    """Write a complete JSON file atomically to avoid half-written results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False, default=_json_default)
        handle.write("\n")
    os.replace(temporary, path)


def environment_record() -> Dict[str, str]:
    """Record dependency versions that affect numerical output."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "psutil": psutil.__version__,
    }


@dataclass
class StageProfile:
    name: str
    seconds: float
    peak_rss_bytes: int
    rss_start_bytes: int
    rss_end_bytes: int


class RunProfiler:
    """Measure wall time and sampled process RSS separately for each stage.

    The RSS peak is a sampled estimate, so very short allocation spikes may be
    missed. The 20 ms polling interval and this limitation are reported with
    each run rather than being represented as an exact allocator peak.
    """

    def __init__(self, interval_seconds: float = 0.02) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds
        self.stages: List[StageProfile] = []
        self._process = psutil.Process()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        stop = threading.Event()
        start_rss = self._process.memory_info().rss
        peak = [start_rss]

        def sample() -> None:
            while not stop.wait(self.interval_seconds):
                try:
                    peak[0] = max(peak[0], self._process.memory_info().rss)
                except psutil.Error:
                    break

        thread = threading.Thread(target=sample, name="rss-sampler", daemon=True)
        start = time.perf_counter()
        thread.start()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            stop.set()
            thread.join(timeout=1.0)
            try:
                end_rss = self._process.memory_info().rss
            except psutil.Error:
                end_rss = peak[0]
            peak[0] = max(peak[0], end_rss)
            self.stages.append(StageProfile(name, elapsed, peak[0],
                                            start_rss, end_rss))

    def record(self) -> Dict[str, Any]:
        return {
            "rss_sampling_interval_seconds": self.interval_seconds,
            "rss_peak_is_sampled": True,
            "stages": [asdict(stage) for stage in self.stages],
            "total_profiled_seconds": sum(s.seconds for s in self.stages),
        }
