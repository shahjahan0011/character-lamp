"""Lightweight JSONL event/metrics logging.

Not a general observability platform -- just enough real instrumentation
to back the technical note's latency/CPU/memory claims with actual
recorded numbers instead of prose estimates. Best-effort: a write or
sampling failure here must never break the demo, so every failure is
swallowed and reported only via on_debug.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

import psutil


class MetricsLog:
    """Appends one JSON object per line to `path`. With `path=None`, every
    call is a no-op -- callers (CharacterOrchestrator) don't need to branch
    on whether metrics collection is enabled."""

    def __init__(self, path: Path | None = None, on_debug=None):
        self._on_debug = on_debug or (lambda msg: None)
        self._process = psutil.Process(os.getpid())
        # Primes psutil's internal CPU-time snapshot; the first real
        # cpu_percent() call after this will report a meaningful delta
        # instead of 0.0 (or a since-process-start average).
        self._process.cpu_percent(interval=None)
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Left open for the object's whole lifetime (appended to across
            # many calls), not a scoped read/write -- a `with` block doesn't
            # fit this usage.
            self._file = open(path, "a", buffering=1)  # noqa: SIM115

    def event(self, kind: str, **fields: Any) -> None:
        if self._file is None:
            return
        record = {"kind": kind, "t_wall": time.time(), "t_monotonic": time.monotonic(), **fields}
        try:
            self._file.write(json.dumps(record) + "\n")
        except OSError as exc:
            self._on_debug(f"metrics: write failed: {exc}")

    def sample_resources(self) -> None:
        try:
            cpu_percent = self._process.cpu_percent(interval=None)
            rss_bytes = self._process.memory_info().rss
        except psutil.Error as exc:
            self._on_debug(f"metrics: resource sample failed: {exc}")
            return
        self.event("resource_sample", cpu_percent=cpu_percent, rss_bytes=rss_bytes)

    def close(self) -> None:
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
