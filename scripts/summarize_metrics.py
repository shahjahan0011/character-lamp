"""Summarizes a metrics.jsonl produced by a run_character.py session.

Usage:
  python scripts/summarize_metrics.py [path/to/metrics.jsonl]

Prints median/p95 first-audio latency, engagement/disengagement counts,
and CPU%/RSS stats over the run -- the real numbers this project's
technical note cites, computed from an actual JSONL log rather than
estimated.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return ordered[idx]


def summarize(path: Path) -> None:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if not events:
        print(f"{path}: no events")
        return

    first_audio = [e["latency_s"] for e in events if e["kind"] == "first_audio"]
    engaged = [e for e in events if e["kind"] == "engaged"]
    disengaged = [e for e in events if e["kind"] == "disengaged"]
    cpu = [e["cpu_percent"] for e in events if e["kind"] == "resource_sample"]
    rss = [e["rss_bytes"] for e in events if e["kind"] == "resource_sample"]

    span_s = events[-1]["t_monotonic"] - events[0]["t_monotonic"]
    print(f"{path}")
    print(f"  events: {len(events)} spanning {span_s:.1f}s")
    print(f"  engagement transitions: {len(engaged)} engaged, {len(disengaged)} disengaged")

    if first_audio:
        print(
            f"  first-audio latency (n={len(first_audio)}): "
            f"median={statistics.median(first_audio):.2f}s "
            f"p95={percentile(first_audio, 95):.2f}s "
            f"min={min(first_audio):.2f}s max={max(first_audio):.2f}s"
        )
    else:
        print("  first-audio latency: no turns recorded")

    if cpu:
        # First sample is unreliable (see MetricsLog's priming comment).
        cpu_samples = cpu[1:] if len(cpu) > 1 else cpu
        print(
            f"  CPU% (n={len(cpu_samples)}, process-relative, one core = 100%): "
            f"mean={statistics.mean(cpu_samples):.1f} max={max(cpu_samples):.1f}"
        )
    if rss:
        print(f"  RSS (n={len(rss)}): mean={statistics.mean(rss) / 1e6:.1f} MB peak={max(rss) / 1e6:.1f} MB")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("var/metrics.jsonl")
    if not target.exists():
        print(f"no such file: {target}")
        sys.exit(1)
    summarize(target)
