#!/usr/bin/env python3
"""Render a dependency-free SVG summary of the completed WBC DDPM baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape


WIDTH, HEIGHT = 1500, 1050
INK = "#233044"
MUTED = "#637083"
GRID = "#dce2ea"
BLUE = "#2878b5"
ORANGE = "#e07a32"
GREEN = "#22856a"
RED = "#bf4e4e"


def txt(x: float, y: float, value: str, size: int = 18, color: str = INK, anchor: str = "start", weight: str = "normal") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans,Arial,sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{escape(value)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, color: str = GRID, width: float = 1.5) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}" />'


def rect(x: float, y: float, w: float, h: float, color: str, opacity: float = 1.0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{color}" fill-opacity="{opacity}" rx="3" />'


def path(points: list[tuple[float, float]], color: str, width: float = 3.0) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round" />'


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def y_log(value: float, low: float, high: float, top: float, height: float) -> float:
    value = min(high, max(low, value))
    return top + (math.log10(high) - math.log10(value)) / (math.log10(high) - math.log10(low)) * height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=Path("/projects/keyi005ssd/WBCDiffusion/results/wbc-raabin-256-baseline-v1/train.csv"))
    parser.add_argument("--validation-csv", type=Path, default=Path("/projects/keyi005ssd/WBCDiffusion/results/wbc-raabin-256-baseline-v1/validation.csv"))
    parser.add_argument("--metrics-json", type=Path, default=Path("/projects/keyi005ssd/WBCDiffusion/results/wbc-raabin-256-baseline-v1/evaluation/fid_kid_step200000_n2000.json"))
    parser.add_argument("--output", type=Path, default=Path("results/wbc_unconditional_baseline_001/diagnostics.svg"))
    args = parser.parse_args()

    train = read_rows(args.train_csv)
    validation = read_rows(args.validation_csv)
    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    smoothed = []
    chunk = 50
    for start in range(0, len(train), chunk):
        values = [float(row["loss"]) for row in train[start : start + chunk]]
        steps = [int(row["step"]) for row in train[start : start + chunk]]
        if values:
            smoothed.append((mean(steps), mean(values)))

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">', rect(0, 0, WIDTH, HEIGHT, "#ffffff")]
    svg.append(txt(65, 45, "WBC DDPM baseline diagnostics", 28, INK, weight="bold"))
    svg.append(txt(65, 72, "Raabin-WBC · 256×256 · EMA step 200,000 · generic Inception-v3 features", 16, MUTED))

    # Panel 1: training/validation objective.
    x0, y0, pw, ph = 120, 125, 1280, 315
    svg.append(txt(x0, y0 - 20, "Denoising objective (log scale)", 20, INK, weight="bold"))
    low, high = 1e-4, 1.5
    for power in (-4, -3, -2, -1, 0):
        value = 10.0**power
        yy = y_log(value, low, high, y0, ph)
        svg.append(line(x0, yy, x0 + pw, yy))
        svg.append(txt(x0 - 12, yy + 5, f"10^{power}", 13, MUTED, "end"))
    for step in (0, 50000, 100000, 150000, 200000):
        xx = x0 + step / 200000 * pw
        svg.append(line(xx, y0, xx, y0 + ph, GRID, 1))
        svg.append(txt(xx, y0 + ph + 26, f"{step//1000}k", 13, MUTED, "middle"))
    train_points = [(x0 + step / 200000 * pw, y_log(loss, low, high, y0, ph)) for step, loss in smoothed if loss > 0]
    val_points = [(x0 + int(row["step"]) / 200000 * pw, y_log(float(row["epsilon_mse"]), low, high, y0, ph)) for row in validation]
    svg.append(path(train_points, BLUE, 2.5))
    svg.append(path(val_points, ORANGE, 3.0))
    svg.append(line(x0 + 20, y0 - 1, x0 + 48, y0 - 1, BLUE, 3))
    svg.append(txt(x0 + 56, y0 + 5, "train loss, 5k-update mean", 14, MUTED))
    svg.append(line(x0 + 300, y0 - 1, x0 + 328, y0 - 1, ORANGE, 3))
    svg.append(txt(x0 + 336, y0 + 5, "fixed-seed validation epsilon MSE", 14, MUTED))
    svg.append(txt(x0 + pw, y0 + ph + 26, "update step", 13, MUTED, "end"))

    # Panel 2: PRDC scores against a real-vs-real reference.
    left, top, width, height = 110, 525, 630, 390
    svg.append(txt(left, top - 28, "PRDC (higher is better)", 20, INK, weight="bold"))
    generated = metrics["prdc_generated_vs_test"]
    real = metrics["prdc_real_test_split_reference"]
    names = (("precision", "Precision"), ("recall", "Recall"), ("density", "Density"), ("coverage", "Coverage"))
    chart_top, chart_h = top + 20, height - 72
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        yy = chart_top + (1 - fraction) * chart_h
        svg.append(line(left + 56, yy, left + width, yy))
        svg.append(txt(left + 44, yy + 5, f"{fraction:.2f}", 12, MUTED, "end"))
    group_w = (width - 92) / len(names)
    bar_w = 32
    for index, (key, label) in enumerate(names):
        cx = left + 74 + (index + 0.5) * group_w
        for offset, value, color in ((-bar_w / 2 - 3, generated[key], BLUE), (bar_w / 2 + 3, real[key], GREEN)):
            bh = value * chart_h
            svg.append(rect(cx + offset - bar_w / 2, chart_top + chart_h - bh, bar_w, bh, color))
            svg.append(txt(cx + offset, chart_top + chart_h - bh - 8, f"{value:.2f}", 11, color, "middle"))
        svg.append(txt(cx, chart_top + chart_h + 28, label, 13, MUTED, "middle"))
    svg.append(rect(left + 85, top + height - 4, 14, 14, BLUE))
    svg.append(txt(left + 106, top + height + 8, "generated vs test", 13, MUTED))
    svg.append(rect(left + 290, top + height - 4, 14, 14, GREEN))
    svg.append(txt(left + 311, top + height + 8, "real test split reference", 13, MUTED))

    # Panel 3: FID and KID point estimates on a common log axis.
    left2, top2, width2, height2 = 820, 525, 570, 390
    svg.append(txt(left2, top2 - 28, "FID / KID (log scale; lower is better)", 20, INK, weight="bold"))
    chart_top2, chart_h2 = top2 + 20, height2 - 72
    lo2, hi2 = 1e-7, 1.0
    for power in (-7, -5, -3, -1, 0):
        value = 10.0**power
        yy = y_log(value, lo2, hi2, chart_top2, chart_h2)
        svg.append(line(left2 + 70, yy, left2 + width2, yy))
        svg.append(txt(left2 + 58, yy + 5, f"1e{power}", 12, MUTED, "end"))
    series = [
        ("FID", metrics["generated_vs_test"]["fid"], metrics["real_train_vs_test_reference"]["fid"]),
        ("KID", metrics["generated_vs_test"]["kid_mean"], metrics["real_train_vs_test_reference"]["kid_mean"]),
    ]
    group_w2 = (width2 - 100) / 2
    for index, (label, gen_v, real_v) in enumerate(series):
        cx = left2 + 80 + (index + 0.5) * group_w2
        for offset, value, color in ((-27, gen_v, RED), (27, real_v, GREEN)):
            yy = y_log(value, lo2, hi2, chart_top2, chart_h2)
            svg.append(rect(cx + offset - 18, yy, 36, chart_top2 + chart_h2 - yy, color))
            svg.append(txt(cx + offset, yy - 8, f"{value:.2g}", 12, color, "middle"))
        svg.append(txt(cx, chart_top2 + chart_h2 + 28, label, 15, MUTED, "middle"))
    svg.append(rect(left2 + 88, top2 + height2 - 4, 14, 14, RED))
    svg.append(txt(left2 + 109, top2 + height2 + 8, "generated vs test", 13, MUTED))
    svg.append(rect(left2 + 300, top2 + height2 - 4, 14, 14, GREEN))
    svg.append(txt(left2 + 321, top2 + height2 + 8, "real train vs test", 13, MUTED))

    svg.append(txt(65, 1005, "Interpret cautiously: Inception-v3 features are generic natural-image features, not validated WBC morphology metrics. Blind human review is pending.", 14, MUTED))
    svg.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
