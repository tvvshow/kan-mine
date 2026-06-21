#!/usr/bin/env python3
"""Parse TC_TIMING logs from bench/sweep_sm86_groupm.sh.

The sweep emits one log per build/runtime combination.  This helper summarizes
all per-draw `tc(cutlass2)` timing lines, not only the final line, so a noisy
single draw does not mislead GROUPM/KSTAGES tuning.
"""

from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path


TC_RE = re.compile(
    r"tc\(cutlass2\): .*?s(?P<kstages>\d+) T\d+ G(?P<groupm>\d+) "
    r"FUSED (?P<tiles>\d+) tiles, prep\+gather=(?P<prep>[0-9.]+) ms, "
    r"search=(?P<search_ms>[0-9.]+) ms (?P<search_ths>[0-9.]+) TH/s, "
    r"total=(?P<total_ms>[0-9.]+) ms (?P<total_ths>[0-9.]+) TH/s"
)
MINE_RE = re.compile(r"MINE done: .*? (?P<ths>[0-9.]+) TH/s")
NAME_RE = re.compile(r"bench_g(?P<groupm>\d+)_k(?P<kstages>\d+)_p(?P<persist>\d+)\.log")


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def parse_log(path: Path) -> dict[str, object] | None:
    text = path.read_text(errors="replace")
    rows = [m.groupdict() for m in TC_RE.finditer(text)]
    if not rows:
        return None

    name = NAME_RE.search(path.name)
    groupm = int(name.group("groupm")) if name else int(rows[-1]["groupm"])
    kstages = int(name.group("kstages")) if name else int(rows[-1]["kstages"])
    persist = int(name.group("persist")) if name else -1

    # Treat the first measured draw as warmup by default when there are enough
    # samples: CUDA context/cache/raster first-use noise should not select a
    # production parameter.
    warmup = 1 if len(rows) >= 4 else 0
    use_rows = rows[warmup:]
    prep = [float(r["prep"]) for r in use_rows]
    search_ms = [float(r["search_ms"]) for r in use_rows]
    search_ths = [float(r["search_ths"]) for r in use_rows]
    total_ms = [float(r["total_ms"]) for r in use_rows]
    total_ths = [float(r["total_ths"]) for r in use_rows]

    mine_match = None
    for mine_match in MINE_RE.finditer(text):
        pass

    return {
        "file": path.name,
        "groupm": groupm,
        "kstages": kstages,
        "persist": persist,
        "samples": len(use_rows),
        "warmup_dropped": warmup,
        "prep_median_ms": median(prep),
        "search_median_ms": median(search_ms),
        "search_avg_ths": mean(search_ths),
        "search_median_ths": median(search_ths),
        "total_median_ms": median(total_ms),
        "total_avg_ths": mean(total_ths),
        "total_median_ths": median(total_ths),
        "mine_done_ths": float(mine_match.group("ths")) if mine_match else 0.0,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: parse_tc_sweep.py <sweep-result-dir>", file=sys.stderr)
        return 2

    outdir = Path(sys.argv[1])
    logs = sorted(outdir.glob("bench_g*_k*_p*.log"))
    parsed = [row for log in logs if (row := parse_log(log)) is not None]
    if not parsed:
        print(f"no TC_TIMING logs found in {outdir}", file=sys.stderr)
        return 1

    parsed.sort(
        key=lambda r: (
            -float(r["total_median_ths"]),
            -float(r["search_median_ths"]),
            int(r["groupm"]),
            int(r["kstages"]),
            int(r["persist"]),
        )
    )

    csv_path = outdir / "tc_sweep_summary.csv"
    fields = [
        "groupm",
        "kstages",
        "persist",
        "samples",
        "warmup_dropped",
        "prep_median_ms",
        "search_median_ms",
        "search_median_ths",
        "search_avg_ths",
        "total_median_ms",
        "total_median_ths",
        "total_avg_ths",
        "mine_done_ths",
        "file",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(parsed)

    print(f"csv={csv_path}")
    print("rank  G  K  P  samples  prep_ms  search_TH/s  total_TH/s  mine_TH/s")
    for i, row in enumerate(parsed[:10], 1):
        print(
            f"{i:>4} {int(row['groupm']):>2} {int(row['kstages']):>2} "
            f"{int(row['persist']):>1} {int(row['samples']):>8} "
            f"{float(row['prep_median_ms']):>8.2f} "
            f"{float(row['search_median_ths']):>11.2f} "
            f"{float(row['total_median_ths']):>10.2f} "
            f"{float(row['mine_done_ths']):>9.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
