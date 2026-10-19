#!/usr/bin/env python3
"""Build native-label class tables and dependency-free SVG bar charts.

Input is a reviewed, source-specific count table. The tool intentionally does
not discover class labels from directory names or harmonize labels implicitly.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
from collections import defaultdict
from pathlib import Path

KEYS = ("source_id", "release", "split", "task", "label_schema", "count_basis")
REQUIRED = set(KEYS) | {"native_label", "count", "data_role"}


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return value or "unnamed"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")
        rows = []
        seen = set()
        for line_no, row in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in row.items()}
            if not row["source_id"] or not row["native_label"]:
                raise ValueError(f"Line {line_no}: source_id and native_label are required")
            try:
                row["count"] = int(row["count"])
            except ValueError as exc:
                raise ValueError(f"Line {line_no}: count must be an integer") from exc
            if row["count"] < 0:
                raise ValueError(f"Line {line_no}: count cannot be negative")
            key = tuple(row.get(k, "") for k in KEYS) + (row["native_label"],)
            if key in seen:
                raise ValueError(f"Line {line_no}: duplicate source/category key {key}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("Input CSV contains no data rows")
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str | int | float]]:
    totals: dict[tuple[str, ...], int] = defaultdict(int)
    for row in rows:
        totals[tuple(row.get(k, "") for k in KEYS)] += row["count"]
    result = []
    for row in rows:
        group = tuple(row.get(k, "") for k in KEYS)
        total = totals[group]
        result.append({
            **row,
            "group_total": total,
            "share_pct": round(100 * row["count"] / total, 4) if total else 0.0,
        })
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_chart(path: Path, group: list[dict]) -> None:
    ordered = sorted(group, key=lambda row: (-row["count"], row["native_label"].casefold()))
    total = ordered[0]["group_total"]
    width, left, right = 940, 235, 95
    top, row_h, bottom = 86, 36, 32
    height = top + row_h * len(ordered) + bottom
    plot_w = width - left - right
    max_count = max((row["count"] for row in ordered), default=0) or 1
    title = " · ".join(str(ordered[0][key]) for key in ("source_id", "release", "split", "task"))
    basis = ordered[0]["count_basis"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#20242b}.title{font-size:20px;font-weight:700}.meta{font-size:13px;fill:#565e6b}.label{font-size:14px}.value{font-size:13px}.axis{stroke:#c8cdd5;stroke-width:1}</style>',
        f'<text class="title" x="{left}" y="32">{html.escape(title)}</text>',
        f'<text class="meta" x="{left}" y="56">N = {total:,} · count basis: {html.escape(basis)}</text>',
        f'<line class="axis" x1="{left}" y1="{top - 9}" x2="{left}" y2="{height - bottom + 3}"/>',
    ]
    for index, row in enumerate(ordered):
        y = top + index * row_h
        bar_w = round(plot_w * row["count"] / max_count)
        label = html.escape(str(row["native_label"]))
        pct = float(row["share_pct"])
        lines.extend([
            f'<text class="label" x="{left - 12}" y="{y + 18}" text-anchor="end">{label}</text>',
            f'<rect x="{left}" y="{y}" width="{bar_w}" height="23" rx="3" fill="#2673b8"/>',
            f'<text class="value" x="{left + bar_w + 9}" y="{y + 17}">{row["count"]:,} ({pct:.2f}%)</text>',
        ])
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counts_csv", type=Path, help="Reviewed native-label count table")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    rows = summarize(read_rows(args.counts_csv))
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    write_csv(args.output / "class_counts_summary.csv", rows, fields)

    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in KEYS)].append(row)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    report = [
        "# Dataset category count report", "",
        "Counts below preserve each source's native label scheme. Rows are not",
        "cross-source comparable unless an explicit harmonization is documented.", "",
    ]
    for group_key, group in sorted(groups.items()):
        title = " · ".join(group_key[:4])
        chart_name = "--".join(slug(part) for part in group_key) + ".svg"
        write_chart(figures / chart_name, group)
        total = group[0]["group_total"]
        report.extend([
            f"## {title}", "",
            f"Count basis: `{group_key[-1]}`; total: **{total:,}**; role: `{group[0]['data_role']}`.", "",
            f"![Native category counts for {title}](figures/{chart_name})", "",
            "| Native category | Count | Share |", "|---|---:|---:|",
        ])
        for row in sorted(group, key=lambda item: (-item["count"], item["native_label"].casefold())):
            report.append(f"| {row['native_label']} | {row['count']:,} | {row['share_pct']:.2f}% |")
        report.append("")
    (args.output / "class_counts_report.md").write_text("\n".join(report), encoding="utf-8")

    mapped = [row for row in rows if row.get("analysis_class") and row.get("harmonization_id")]
    if mapped:
        totals: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
        for row in mapped:
            key = (row["harmonization_id"], row["data_role"], row["split"],
                   row["source_id"], row["analysis_class"])
            totals[key] += row["count"]
        labels = sorted({key[-1] for key in totals}, key=str.casefold)
        row_totals: dict[tuple[str, str, str, str], int] = defaultdict(int)
        wide: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(dict)
        for (hid, role, split, source_id, label), count in totals.items():
            row_key = (hid, role, split, source_id)
            wide[row_key][label] = count
            row_totals[row_key] += count
        count_rows = []
        share_rows = []
        for row_key in sorted(wide):
            hid, role, split, source_id = row_key
            total = row_totals[row_key]
            base = {"harmonization_id": hid, "data_role": role, "split": split,
                    "source_id": source_id}
            count_rows.append({**base, **wide[row_key], "total": total})
            share_rows.append({**base,
                               **{label: round(100 * wide[row_key].get(label, 0) / total, 4)
                                  if total else 0.0 for label in labels},
                               "total_share_pct": 100.0 if total else 0.0})
        count_fields = ["harmonization_id", "data_role", "split", "source_id", *labels, "total"]
        share_fields = ["harmonization_id", "data_role", "split", "source_id", *labels,
                        "total_share_pct"]
        write_csv(args.output / "source_by_harmonized_class_counts.csv", count_rows, count_fields)
        write_csv(args.output / "source_by_harmonized_class_shares_pct.csv", share_rows, share_fields)


if __name__ == "__main__":
    main()
