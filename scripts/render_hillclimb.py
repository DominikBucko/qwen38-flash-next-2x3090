#!/usr/bin/env python3
"""Render the README hillclimb chart with the Python standard library."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


WIDTH = 1600
HEIGHT = 1000
INK = "#172033"
MUTED = "#637083"
GRID = "#dce3ec"
BLUE = "#2563eb"
BLUE_LIGHT = "#dbeafe"
GOLD = "#c58a16"
PANEL = "#f8fafc"


def text(x: float, y: float, value: str, size: int = 22, *, anchor: str = "start",
         weight: int = 400, fill: str = INK) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{html.escape(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = GRID,
         width: float = 2, dash: str | None = None) -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{extra}/>'
    )


def panel(parts: list[str], x: int, y: int, w: int, h: int, title: str,
          subtitle: str) -> None:
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" '
        f'fill="{PANEL}" stroke="{GRID}" stroke-width="2"/>'
    )
    parts.append(text(x + 30, y + 44, title, 27, weight=700))
    parts.append(text(x + 30, y + 76, subtitle, 17, fill=MUTED))


def render(data: dict) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img">',
        "<title>Qwen3.8-Flash-Next performance hillclimb</title>",
        "<desc>Three charts show matched decode tuning, MTP long-decode gains, and selected prefill records on two RTX 3090 GPUs with 128 GB of memory.</desc>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        text(70, 66, "Qwen3.8-Flash-Next performance hillclimb", 40, weight=750),
        text(70, 101, "2 × RTX 3090 (24 GB) · 128 GB system memory · single-request measurements", 20, fill=MUTED),
    ]

    # Main matched-decode progression.
    panel(parts, 60, 130, 1480, 470, "Decode hillclimb to 135 tok/s",
          "Matched through LRU-104; dotted break changes output from 256 to 4,096 tokens")
    matched_points = data["matched_decode"]["points"]
    long_points = [
        {
            "name": point["name"],
            "short_name": [point["name"].replace(" ", "\n", 1).split("\n")[0],
                           point["name"].replace(" ", "\n", 1).split("\n")[1]],
            "tokens_per_second": point["tokens_per_second"],
        }
        for point in data["long_decode"]["points"][1:]
    ]
    points = matched_points + long_points
    left, right, top, bottom = 125, 1500, 245, 495
    y_min, y_max = 20.0, 145.0
    y_of = lambda value: bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
    x_of = lambda i: left + i * (right - left) / (len(points) - 1)
    for tick in (25, 50, 75, 100, 125):
        y = y_of(tick)
        parts.append(line(left, y, right, y))
        parts.append(text(left - 16, y + 7, str(tick), 17, anchor="end", fill=MUTED))
    parts.append(text(left - 16, top - 20, "tok/s", 16, anchor="end", fill=MUTED))
    coords = [(x_of(i), y_of(point["tokens_per_second"])) for i, point in enumerate(points)]
    matched_coords = coords[:len(matched_points)]
    long_coords = coords[len(matched_points):]
    area = " ".join(f"{x:.1f},{y:.1f}" for x, y in matched_coords)
    parts.append(
        f'<polygon points="{left},{bottom} {area} {matched_coords[-1][0]:.1f},{bottom}" '
        f'fill="{BLUE_LIGHT}" opacity="0.72"/>'
    )
    parts.append(
        '<polyline points="' + area + f'" fill="none" stroke="{BLUE}" '
        'stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    bridge_start = matched_coords[-1]
    bridge_end = long_coords[0]
    parts.append(line(bridge_start[0], bridge_start[1], bridge_end[0], bridge_end[1],
                      stroke=MUTED, width=4, dash="10 10"))
    long_path = " ".join(f"{x:.1f},{y:.1f}" for x, y in long_coords)
    parts.append(
        '<polyline points="' + long_path + f'" fill="none" stroke="{GOLD}" '
        'stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    boundary_x = (bridge_start[0] + bridge_end[0]) / 2
    parts.append(line(boundary_x, top - 10, boundary_x, bottom + 4,
                      stroke=MUTED, width=2, dash="5 7"))
    parts.append(text(boundary_x + 10, top + 18, "4K output workload", 15,
                      fill=MUTED))
    for i, (point, (x, y)) in enumerate(zip(points, coords)):
        long_phase = i >= len(matched_points)
        fill = GOLD if long_phase else "#ffffff"
        stroke = GOLD if long_phase else BLUE
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="5"/>'
        )
        parts.append(text(x, y - 20, f'{point["tokens_per_second"]:.1f}', 18,
                          anchor="middle", weight=700, fill=INK))
        for row, label in enumerate(point["short_name"]):
            parts.append(text(x, bottom + 40 + row * 22, label, 15,
                              anchor="middle", fill=MUTED))
    parts.append(text(1495, 208, "135.2 tok/s peak", 25, anchor="end", weight=750, fill=GOLD))

    # Lower-left MTP bars.
    panel(parts, 60, 630, 710, 315, "MTP unlocks long decode",
          "128 input → 4,096 output; warmed run")
    mtp = data["long_decode"]["points"]
    left, base, chart_top = 125, 885, 745
    bar_w, gap = 150, 55
    for tick in (0, 50, 100, 150):
        y = base - tick / 150 * (base - chart_top)
        parts.append(line(left, y, 730, y))
        parts.append(text(left - 14, y + 6, str(tick), 15, anchor="end", fill=MUTED))
    for i, point in enumerate(mtp):
        x = left + 55 + i * (bar_w + gap)
        h = point["tokens_per_second"] / 150 * (base - chart_top)
        color = BLUE if i else "#94a3b8"
        if i == len(mtp) - 1:
            color = GOLD
        parts.append(f'<rect x="{x}" y="{base-h:.1f}" width="{bar_w}" height="{h:.1f}" rx="8" fill="{color}"/>')
        parts.append(text(x + bar_w / 2, base - h - 12,
                          f'{point["tokens_per_second"]:.1f}', 19,
                          anchor="middle", weight=700))
        parts.append(text(x + bar_w / 2, base + 30, point["name"], 16,
                          anchor="middle", fill=MUTED))

    # Lower-right prefill bars.
    panel(parts, 830, 630, 710, 315, "Selected prefill records",
          "Prompt throughput; context and cache mode are named below")
    prefill = data["prefill_records"]["points"]
    left, base, chart_top = 895, 885, 745
    for tick in (0, 500, 1000, 1500):
        y = base - tick / 1750 * (base - chart_top)
        parts.append(line(left, y, 1500, y))
        parts.append(text(left - 14, y + 6, f"{tick:,}", 15, anchor="end", fill=MUTED))
    for i, point in enumerate(prefill):
        x = left + 55 + i * (bar_w + gap)
        h = point["tokens_per_second"] / 1750 * (base - chart_top)
        color = GOLD if "record" in point["name"] else BLUE
        parts.append(f'<rect x="{x}" y="{base-h:.1f}" width="{bar_w}" height="{h:.1f}" rx="8" fill="{color}"/>')
        parts.append(text(x + bar_w / 2, base - h - 12,
                          f'{point["tokens_per_second"]:,.0f}', 19,
                          anchor="middle", weight=700))
        label = point["name"].replace(" ", "\n", 1)
        for row, word in enumerate(label.split("\n")):
            parts.append(text(x + bar_w / 2, base + 28 + row * 20, word, 15,
                              anchor="middle", fill=MUTED))

    parts.append(text(60, 982,
                      "The blue line uses one workload. The dotted break and gold line use a longer 4K output.",
                      16, fill=MUTED))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("benchmarks/hillclimb.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/images/hillclimb.svg"))
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data))
    print(args.output)


if __name__ == "__main__":
    main()
