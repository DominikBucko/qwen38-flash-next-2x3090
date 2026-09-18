#!/usr/bin/env python3
"""Export the September 18 prefill curves, using the existing SVG house style."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

# The repository validator rejects generated Python cache files.
sys.dont_write_bytecode = True
from render_hillclimb import BLUE, GOLD, INK, MUTED, line, text

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/2026-09-18/summary.json"
WIDTH = 1120


def start(height: int, title: str, description: str, source_hash: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{title}</title>",
        f"<desc id=\"desc\">{description}</desc>",
        f"<!-- source-json-sha256: {source_hash} -->",
        f'<rect width="{WIDTH}" height="{height}" fill="#ffffff"/>',
    ]


def curve(parts: list[str], coords: list[tuple[float, float]], color: str,
          shape: str = "circle", dash: str | None = None) -> None:
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                 f'stroke-width="3"{extra}/>')
    for x, y in coords:
        if shape == "square":
            parts.append(f'<rect x="{x-5:.2f}" y="{y-5:.2f}" width="10" '
                         f'height="10" fill="white" stroke="{color}" stroke-width="2.5"/>')
        elif shape == "triangle":
            parts.append(f'<polygon points="{x:.2f},{y-7:.2f} {x-6:.2f},{y+5:.2f} '
                         f'{x+6:.2f},{y+5:.2f}" fill="{color}"/>')
        else:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" '
                         f'fill="{color}"/>')


def axes(parts: list[str], left: int, right: int, top: int, bottom: int,
         ymax: float, ticks: list[float], unit: str):
    def y_of(value: float) -> float:
        return bottom - value / ymax * (bottom - top)
    for tick in ticks:
        y = y_of(tick)
        parts.append(line(left, y, right, y, width=1))
        parts.append(text(left - 16, y + 6, f"{tick:,.0f}", 19,
                          anchor="end", fill=MUTED))
    parts.append(line(left, top, left, bottom, stroke=MUTED, width=1))
    parts.append(line(left, bottom, right, bottom, stroke=MUTED, width=1))
    parts.append(text(left, top - 18, unit, 18, fill=MUTED))
    return y_of


def long_context(data: dict, source_hash: str) -> str:
    series = data["long_context_series"]
    first = series[0]["points"][-1]["ttft_seconds"]
    last = series[-1]["points"][-1]["ttft_seconds"]
    title = f"Full-context first token: {first:.1f} s → {last:.1f} s"
    parts = start(670, title, "Single measured requests at 131,072 and 260,096 input "
                  "tokens, each with 2,048 output tokens. Lower time is better. "
                  "Experimental builds; preceding warm state differs.", source_hash)
    parts += [text(65, 56, title, 34, weight=700),
              text(65, 92, "2 × RTX 3090 · 128 GB RAM · experimental runtime screens", 21, fill=MUTED)]
    colors = [MUTED, BLUE, GOLD]
    shapes = ["square", "triangle", "circle"]
    dashes = ["8 6", "3 5", None]
    for i, item in enumerate(series):
        x = 65 + 345 * i
        curve(parts, [(x, 131), (x+30, 131)], colors[i], shapes[i], dashes[i])
        parts.append(text(x+43, 138, item["label"], 19))
    left, right, top, bottom = 112, 993, 198, 503
    y_of = axes(parts, left, right, top, bottom, 240, [0, 60, 120, 180, 240],
                "Seconds to first token ↓")
    x_of = lambda value: left + (value - 131072) / (260096 - 131072) * (right - left)
    for i, item in enumerate(series):
        coords = [(x_of(p["input_tokens"]), y_of(p["ttft_seconds"])) for p in item["points"]]
        curve(parts, coords, colors[i], shapes[i], dashes[i])
        if i in (0, 2):
            for j, (x, y) in enumerate(coords):
                # Put labels inside the plot, away from nearby measured points.
                parts.append(text(x + (13 if j == 0 else -13), y - (14 if i == 0 else 22),
                                  f'{item["points"][j]["ttft_seconds"]:.1f} s', 22,
                                  anchor="start" if j == 0 else "end", weight=700, fill=colors[i]))
    for value in (131072, 260096):
        x = x_of(value)
        parts.append(line(x, bottom, x, bottom+7, stroke=MUTED))
        parts.append(text(x, bottom+35, f"{value:,}", 21, anchor="middle"))
    parts += [text(550, 568, "Input tokens (+ 2,048 output tokens per request)", 21, anchor="middle"),
              text(65, 612, "One request per point; different preceding cache states. Not a repeated one-change A/B.", 19, fill=MUTED),
              text(65, 644, "Lines join measured points only. Source: benchmarks/2026-09-18/summary.json", 18, fill=MUTED),
              "</svg>"]
    return "\n".join(parts) + "\n"


def agent_cache(data: dict, source_hash: str) -> str:
    agent = data["fresh_agent"]
    rows = agent["request_rows"]
    rate = sum(r["new_tokens"] for r in rows) / sum(r["prefill_seconds"] for r in rows)
    reused = 100 * agent["cached_tokens"] / agent["prompt_tokens"]
    parts = start(945, "Fresh-agent prefill and prefix reuse",
                  "One completed task, thirteen requests on the preceding tiered and warmup candidate. "
                  "New prefill throughput excludes cache hits. Context grows to 30,084 tokens.", source_hash)
    parts += [text(65, 56, "Fresh agent: compute only the new part", 34, weight=700),
              text(65, 92, "Preceding tiered + warmup candidate · one completed task · 13 requests", 21, fill=MUTED),
              text(65, 141, f"{rate:,.0f} new tok/s aggregate · {reused:.0f}% prompt tokens reused", 26, weight=700)]
    left, right = 112, 1020
    x_of = lambda request: left + (request - 1) / 12 * (right - left)
    y_of = axes(parts, left, right, 206, 404, 2000, [0, 500, 1000, 1500, 2000], "New prefill tokens/s")
    parts.append(line(left, y_of(1500), right, y_of(1500), stroke=INK, width=1.5, dash="7 6"))
    parts.append(text(right, y_of(1500)-13, "1,500 target", 18, anchor="end", fill=MUTED))
    curve(parts, [(x_of(r["request"]), y_of(r["new_tokens"] / r["prefill_seconds"])) for r in rows], BLUE)
    for tick in (1, 4, 7, 10, 13):
        parts.append(text(x_of(tick), 434, str(tick), 19, anchor="middle"))
    parts.append(text(565, 468, "Request number", 20, anchor="middle", fill=MUTED))
    for x, label, color, shape, dash in [(112, "Total prompt", MUTED, "square", "8 6"),
                                       (385, "Newly computed", BLUE, "circle", None)]:
        curve(parts, [(x, 516), (x+30, 516)], color, shape, dash)
        parts.append(text(x+43, 523, label, 20))
    y_of = axes(parts, left, right, 571, 794, 32000, [0, 8000, 16000, 24000, 32000], "Prompt tokens")
    curve(parts, [(x_of(r["request"]), y_of(r["prompt_tokens"])) for r in rows], MUTED, "square", "8 6")
    curve(parts, [(x_of(r["request"]), y_of(r["new_tokens"])) for r in rows], BLUE)
    for tick in (1, 4, 7, 10, 13):
        parts.append(text(x_of(tick), 824, str(tick), 19, anchor="middle"))
    parts += [text(565, 858, "Request number", 20, anchor="middle", fill=MUTED),
              text(65, 904, "One fresh smoke, not a suite rerun. The latest overlap candidate has no completed fresh task.", 19, fill=MUTED),
              text(65, 932, "Source: benchmarks/2026-09-18/summary.json · aggregate = total new tokens / total prefill time", 18, fill=MUTED),
              "</svg>"]
    return "\n".join(parts) + "\n"


def validate(data: dict) -> None:
    for series in data["long_context_series"]:
        assert [p["input_tokens"] for p in series["points"]] == [131072, 260096]
        for point in series["points"]:
            assert point["output_tokens"] == 2048
            assert 0 < point["ttft_seconds"] < point["request_seconds"]
    a = data["fresh_agent"]
    rows = a["request_rows"]
    assert len(rows) == a["requests"] == 13
    assert [r["request"] for r in rows] == list(range(1, 14))
    assert sum(r["prompt_tokens"] for r in rows) == a["prompt_tokens"]
    assert sum(r["new_tokens"] for r in rows) == a["new_prefill_tokens"]
    assert a["prompt_tokens"] - a["new_prefill_tokens"] == a["cached_tokens"]
    seconds = sum(r["prefill_seconds"] for r in rows)
    assert math.isclose(seconds, a["prefill_seconds"])
    assert math.isclose(a["new_prefill_tokens"] / seconds, a["new_prefill_tps"])


def main() -> None:
    raw = SOURCE.read_bytes()
    data = json.loads(raw)
    validate(data)
    source_hash = hashlib.sha256(raw).hexdigest()
    for filename, renderer in [("prefill-long-context.svg", long_context),
                               ("prefill-agent-cache.svg", agent_cache)]:
        output = ROOT / "docs/images" / filename
        output.write_text(renderer(data, source_hash), encoding="utf-8")
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
