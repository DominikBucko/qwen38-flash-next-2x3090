#!/usr/bin/env python3
"""Render API-observed cumulative decode rates from a benchmark report."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GATES = (128, 256, 512, 1024, 2048, 4096)


@dataclass(frozen=True)
class Point:
    emitted_tokens: int
    tokens_per_second: float


@dataclass(frozen=True)
class Series:
    run_index: int
    points: tuple[Point, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a deterministic long-context decode SVG."
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Report JSON or JSON.gz"
    )
    parser.add_argument("--output", required=True, type=Path, help="Output SVG")
    return parser.parse_args()


def load_report(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        encoded = path.read_bytes()
        raw = gzip.decompress(encoded) if path.suffix.lower() == ".gz" else encoded
        report = json.loads(raw)
    except (OSError, EOFError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read benchmark report {path}: {error}") from error
    if not isinstance(report, dict):
        raise ValueError("benchmark report must be a JSON object")
    return report, raw


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def derive_series(
    run: dict[str, Any],
    gates: tuple[int, ...],
    expected_output_tokens: int,
) -> Series:
    run_index = run.get("index")
    if isinstance(run_index, bool) or not isinstance(run_index, int):
        raise ValueError("each measured run must have an integer index")
    chunks = run.get("sse_chunks")
    if not isinstance(chunks, list):
        raise ValueError(f"measured run {run_index} has no SSE chunks")

    cumulative = 0
    first_count: int | None = None
    first_elapsed: float | None = None
    previous_elapsed = -math.inf
    gate_index = 0
    points: list[Point] = []
    for chunk_index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"run {run_index} SSE chunk {chunk_index} is not an object"
            )
        count = chunk.get("output_token_ids_count")
        elapsed = chunk.get("elapsed_seconds")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"run {run_index} SSE chunk {chunk_index} has invalid count"
            )
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed):
            raise ValueError(
                f"run {run_index} SSE chunk {chunk_index} has invalid time"
            )
        elapsed = float(elapsed)
        if elapsed < previous_elapsed:
            raise ValueError(f"run {run_index} SSE elapsed times are not monotonic")
        previous_elapsed = elapsed
        if count == 0:
            continue
        cumulative += count
        if first_count is None:
            first_count = count
            first_elapsed = elapsed
            if first_count >= gates[0]:
                raise ValueError(
                    f"run {run_index} first output chunk already reaches a rate gate"
                )
            continue
        assert first_elapsed is not None
        while gate_index < len(gates) and cumulative >= gates[gate_index]:
            duration = elapsed - first_elapsed
            if duration <= 0:
                raise ValueError(
                    f"run {run_index} reaches a gate without elapsed decode time"
                )
            points.append(
                Point(
                    emitted_tokens=cumulative,
                    tokens_per_second=(cumulative - first_count) / duration,
                )
            )
            gate_index += 1

    if first_count is None:
        raise ValueError(f"measured run {run_index} contains no output token IDs")
    if gate_index != len(gates):
        raise ValueError(
            f"measured run {run_index} emitted {cumulative} tokens; "
            f"it did not reach the {gates[gate_index]}-token gate"
        )
    if cumulative != expected_output_tokens:
        raise ValueError(
            f"measured run {run_index} SSE chunks contain {cumulative} output tokens; "
            f"usage declares {expected_output_tokens}"
        )
    return Series(run_index=run_index, points=tuple(points))


def validate_and_derive(
    report: dict[str, Any],
) -> tuple[list[Series], int, int, str, str]:
    protocol = report.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("benchmark report has no protocol object")
    input_tokens = positive_int(protocol.get("input_tokens"), "protocol.input_tokens")
    output_tokens = positive_int(
        protocol.get("output_tokens"), "protocol.output_tokens"
    )
    if output_tokens != GATES[-1]:
        raise ValueError(
            f"report requests {output_tokens} output tokens; exactly 4096 are required"
        )
    if protocol.get("measured_runs") != 3:
        raise ValueError("protocol.measured_runs must be exactly 3")

    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ValueError("benchmark report has no runs array")
    measured = [
        run for run in runs if isinstance(run, dict) and run.get("phase") == "measured"
    ]
    if len(measured) != 3:
        raise ValueError(f"expected exactly 3 measured runs, found {len(measured)}")
    for run in measured:
        if run.get("status") != "ok":
            raise ValueError(f"measured run {run.get('index')} is not valid")
        usage = run.get("usage")
        if not isinstance(usage, dict):
            raise ValueError(f"measured run {run.get('index')} has no usage object")
        if usage.get("prompt_tokens") != input_tokens:
            raise ValueError(
                f"measured run {run.get('index')} prompt count differs from protocol"
            )
        if usage.get("completion_tokens") != output_tokens:
            raise ValueError(
                f"measured run {run.get('index')} completion count differs from protocol"
            )

    prompt_recipe = report.get("prompt_recipe")
    if not isinstance(prompt_recipe, dict) or not isinstance(
        prompt_recipe.get("style"), str
    ):
        raise ValueError("benchmark report has no prompt recipe style")
    recipe = prompt_recipe["style"]
    method = prompt_recipe.get("method")
    if (
        recipe != "repo-chat"
        and isinstance(method, str)
        and "server-tokenized seed" in method
    ):
        recipe += ", repeated server-tokenized seed"

    declared = report.get("declared_environment")
    if not isinstance(declared, dict) or not isinstance(
        declared.get("server_settings_note"), str
    ):
        raise ValueError("benchmark report has no declared server settings")
    profile = declared["server_settings_note"].replace(";", " · ")
    for recorded, label in {
        "vendor+publicoverlay": "vendor + public overlay",
        "hot84": "hot cache 84",
        "customallreduceenabled": "custom all-reduce enabled",
        "expandable_segmentsFalse": "expandable_segments:False",
        "driverP2Penabled": "driver P2P enabled",
        "262144context": "262,144 context",
        "BF16KV": "BF16 KV",
        "approximateQSA": "approximate QSA",
    }.items():
        profile = profile.replace(recorded, label)
    series = sorted(
        (derive_series(run, GATES, output_tokens) for run in measured),
        key=lambda item: item.run_index,
    )
    if len({item.run_index for item in series}) != len(series):
        raise ValueError("measured run indexes must be distinct")
    return series, input_tokens, output_tokens, recipe, profile


def build_figure(
    series: list[Series],
    input_tokens: int,
    output_tokens: int,
    recipe: str,
    profile: str,
    source_name: str,
) -> Any:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError as error:
        raise RuntimeError("Matplotlib is required to render the SVG") from error

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#64748B",
            "axes.labelcolor": "#273444",
            "axes.linewidth": 0.8,
            "figure.facecolor": "#FBFCFE",
            "axes.facecolor": "#FBFCFE",
            "grid.color": "#DDE4EC",
            "grid.linewidth": 0.8,
            "svg.fonttype": "none",
            "svg.hashsalt": "qwen38-public-decode-v1",
            "text.color": "#17212B",
            "xtick.color": "#536273",
            "ytick.color": "#536273",
        }
    )

    colors = ("#1F4E79", "#397DB5", "#75A9D2")
    line_styles = ("-", (0, (6, 3)), (0, (1.5, 2.2)))
    markers = ("o", "s", "^")
    figure, axis = plt.subplots(figsize=(10.8, 6.6))
    figure.subplots_adjust(left=0.095, right=0.965, top=0.76, bottom=0.225)

    title = "Long-context decode across a 4,096-token output"
    subtitle = (
        f"{input_tokens:,} input tokens · {output_tokens:,} output tokens · "
        f"{len(series)} valid measured runs · recipe: {recipe}"
    )
    profile_note = textwrap.fill(f"Declared serving profile: {profile}", width=132)
    figure.text(
        0.095, 0.945, title, fontsize=18, fontweight="bold", ha="left", va="top"
    )
    figure.text(
        0.095, 0.895, subtitle, fontsize=10.5, ha="left", va="top", color="#394A5A"
    )
    figure.text(
        0.095, 0.858, profile_note, fontsize=9.2, ha="left", va="top", color="#64748B"
    )

    all_rates = [point.tokens_per_second for item in series for point in item.points]
    y_max = max(all_rates)
    final_rates = [item.points[-1].tokens_per_second for item in series]
    collision_gap = max(2.5, y_max * 0.035)
    direct_labels_fit = all(
        right - left >= collision_gap
        for left, right in zip(sorted(final_rates), sorted(final_rates)[1:])
    )
    for position, item in enumerate(series):
        xs = [point.emitted_tokens for point in item.points]
        ys = [point.tokens_per_second for point in item.points]
        label = f"Run {item.run_index + 1} · final {ys[-1]:.2f} tok/s"
        axis.plot(
            xs,
            ys,
            color=colors[position],
            linestyle=line_styles[position],
            linewidth=2.1,
            marker=markers[position],
            markersize=5.8,
            markerfacecolor="#FBFCFE" if position else colors[position],
            markeredgecolor=colors[position],
            markeredgewidth=1.35,
            label=label,
            zorder=3,
        )
        if direct_labels_fit:
            axis.annotate(
                f"Run {item.run_index + 1}  {ys[-1]:.2f}",
                (xs[-1], ys[-1]),
                xytext=(10, 0),
                textcoords="offset points",
                color=colors[position],
                fontsize=9,
                fontweight="bold",
                va="center",
            )

    axis.set_xscale("log", base=2)
    axis.set_xlim(
        GATES[0] / 1.15, output_tokens * (1.22 if direct_labels_fit else 1.06)
    )
    axis.set_ylim(0, y_max * 1.13)
    axis.set_xticks(GATES)
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:,.0f}")
    )
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:,.0f}")
    )
    axis.set_xlabel(
        "Output tokens emitted (log₂ scale; actual cumulative SSE count)",
        labelpad=10,
    )
    axis.set_ylabel("Cumulative API-observed output rate (tokens/s)", labelpad=10)
    axis.grid(axis="y")
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=3.5, width=0.8)
    if not direct_labels_fit:
        legend = axis.legend(
            loc="upper right",
            frameon=True,
            facecolor="#FBFCFE",
            edgecolor="#CAD4DF",
            framealpha=0.97,
            fontsize=9,
        )
        legend.get_frame().set_linewidth(0.8)

    caption = (
        "API-observed cumulative decode rate = tokens after the first output chunk "
        "÷ elapsed time after that chunk.\n"
        "Each point is the first SSE chunk reaching a gate; x shows that chunk’s "
        "actual emitted count. No cross-run mean.\n"
        "All completion tokens count, including reasoning. Forced output length is not an answer-quality score."
    )
    figure.text(
        0.095,
        0.13,
        caption,
        fontsize=9,
        ha="left",
        va="top",
        color="#536273",
        linespacing=1.45,
    )
    figure.text(
        0.095,
        0.025,
        f"Source: {source_name}",
        fontsize=8.5,
        ha="left",
        va="bottom",
        color="#788695",
    )

    return figure


def render_svg(
    series: list[Series],
    input_tokens: int,
    output_tokens: int,
    recipe: str,
    profile: str,
    source_name: str,
) -> str:
    figure = build_figure(
        series,
        input_tokens,
        output_tokens,
        recipe,
        profile,
        source_name,
    )
    buffer = io.StringIO()
    try:
        figure.savefig(
            buffer,
            format="svg",
            metadata={"Creator": "scripts/render_public_decode.py", "Date": None},
        )
        return "\n".join(line.rstrip() for line in buffer.getvalue().splitlines()) + "\n"
    finally:
        figure.clear()


def embed_provenance(svg: str, source_hash: str, generator_hash: str) -> str:
    comments = (
        f"<!-- source-json-sha256: {source_hash} -->\n"
        f"<!-- generator-file-sha256: {generator_hash} -->\n"
    )
    marker = "<svg "
    if marker not in svg:
        raise ValueError("Matplotlib output did not contain an SVG root")
    return svg.replace(marker, comments + marker, 1)


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() != ".svg":
        raise SystemExit("error: --output must end in .svg")
    try:
        report, raw_json = load_report(args.input)
        series, input_tokens, output_tokens, recipe, profile = validate_and_derive(
            report
        )
        svg = render_svg(
            series,
            input_tokens,
            output_tokens,
            recipe,
            profile,
            args.input.name,
        )
        svg = embed_provenance(
            svg,
            hashlib.sha256(raw_json).hexdigest(),
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(svg)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
