#!/usr/bin/env python3
"""Export the September 25 fast-runtime chart from the checked-in measurements."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

# The repository validator rejects generated Python cache files.
sys.dont_write_bytecode = True
from render_hillclimb import BLUE, GOLD, INK, MUTED, line, text
from render_prefill_progress import WIDTH, curve

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "benchmarks/2026-09-18/summary.json"
CURRENT = ROOT / "benchmarks/2026-09-25/summary.json"
OUTPUT = ROOT / "docs/images/fast-256k-progress.svg"
GREEN = "#15803d"
SHAPES = (131072, 260096)


def series(previous: dict, current: dict) -> list[dict]:
    rows = []
    for item in previous["long_context_series"]:
        if item["id"] in ("before_tiered_prefill", "latest_overlap"):
            rows.append({"label": "Sep 18: " + item["label"].split(",")[0],
                         "points": [(p["input_tokens"], p["ttft_seconds"], p["decode_tps"])
                                    for p in item["points"]]})
    release = current["release_image_validation"]
    points = []
    for tokens in SHAPES:
        runs = release[f"input_{tokens}_output_2048"]["measured_runs"]
        assert len(runs) == 3
        points.append((tokens, statistics.median(r["ttft_seconds"] for r in runs),
                       statistics.median(r["decode_tps"] for r in runs)))
    rows.append({"label": "Sep 25 release, median of 3", "points": points})
    return rows


def render(previous: dict, current: dict, hashes: tuple[str, str]) -> str:
    rows = series(previous, current)
    height = 760
    colors, shapes, dashes = [MUTED, GOLD, GREEN], ["square", "triangle", "circle"], ["8 6", "3 5", None]
    last = rows[-1]["points"]
    title = f"Full 256K window: {last[1][1]:.1f} s to first token, {last[1][2]:.1f} tok/s decode"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{title}</title>",
        "<desc id=\"desc\">Seconds to first token and decode tokens per second at 131,072 and "
        "260,096 input tokens, each with 2,048 output tokens. September 18 development screens "
        "versus the median of three September 25 release-image runs.</desc>",
        f"<!-- source-json-sha256: {hashes[0]} -->",
        f"<!-- source-json-sha256: {hashes[1]} -->",
        f'<rect width="{WIDTH}" height="{height}" fill="#ffffff"/>',
        text(65, 56, title, 32, weight=700),
        text(65, 92, "2 × RTX 3090 · 128 GB RAM · 2,048 output tokens per request", 21, fill=MUTED),
    ]
    for i, item in enumerate(rows):
        x = 65 + 345 * i
        curve(parts, [(x, 131), (x + 30, 131)], colors[i], shapes[i], dashes[i])
        parts.append(text(x + 43, 138, item["label"], 19))
    panels = [(112, 520, "Seconds to first token ↓", 240, [0, 60, 120, 180, 240], 1),
              (660, 1068, "Decode tok/s ↑", 120, [0, 30, 60, 90, 120], 2)]
    top, bottom = 205, 540
    for left, right, unit, ymax, ticks, column in panels:
        y_of = lambda value, ymax=ymax: bottom - value / ymax * (bottom - top)
        for tick in ticks:
            parts.append(line(left, y_of(tick), right, y_of(tick), width=1))
            parts.append(text(left - 14, y_of(tick) + 6, f"{tick:,.0f}", 18, anchor="end", fill=MUTED))
        parts.append(line(left, bottom, right, bottom, stroke=MUTED, width=1))
        parts.append(text(left, top - 18, unit, 18, fill=MUTED))
        x_of = lambda tokens, left=left, right=right: left + 40 + (tokens - SHAPES[0]) / (SHAPES[1] - SHAPES[0]) * (right - left - 80)
        for i, item in enumerate(rows):
            coords = [(x_of(p[0]), y_of(p[column])) for p in item["points"]]
            curve(parts, coords, colors[i], shapes[i], dashes[i])
            if i == len(rows) - 1:
                for (x, y), p in zip(coords, item["points"]):
                    label = f"{p[column]:.1f} s" if column == 1 else f"{p[column]:.1f}"
                    parts.append(text(x, y - 16, label, 21, anchor="middle", weight=700, fill=GREEN))
        for tokens in SHAPES:
            parts.append(text(x_of(tokens), bottom + 32, f"{tokens:,}", 19, anchor="middle"))
        parts.append(text((left + right) / 2, bottom + 64, "Input tokens", 19, anchor="middle", fill=MUTED))
    parts += [text(65, 668, "September 18 points are single development screens; September 25 points are the median of three runs.", 18, fill=MUTED),
              text(65, 698, "Lines join measured points only. Sources: benchmarks/2026-09-18/summary.json, benchmarks/2026-09-25/summary.json", 17, fill=MUTED),
              "</svg>"]
    return "\n".join(parts) + "\n"


def main() -> None:
    raw_previous, raw_current = PREVIOUS.read_bytes(), CURRENT.read_bytes()
    hashes = (hashlib.sha256(raw_previous).hexdigest(), hashlib.sha256(raw_current).hexdigest())
    OUTPUT.write_text(render(json.loads(raw_previous), json.loads(raw_current), hashes), encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
