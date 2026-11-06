"""Summarize completed blinded WBC ratings against the preregistered gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


RATING_FIELDS = (
    "wbc_recognizability_1to5",
    "single_cell_completeness_1to5",
    "nucleus_cytoplasm_structure_1to5",
    "stain_background_1to5",
    "artifact_severity_1to5",
)
BOOLEAN_FIELDS = ("repeated_template_yes_no",)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def rating(value: str, field: str, reviewer: str, image_id: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{reviewer}/{image_id}: {field} must be an integer from 1 to 5") from error
    if parsed not in range(1, 6):
        raise ValueError(f"{reviewer}/{image_id}: {field} must be an integer from 1 to 5")
    return parsed


def yes_no(value: str, reviewer: str, image_id: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"yes", "no"}:
        raise ValueError(f"{reviewer}/{image_id}: repeated_template_yes_no must be yes or no")
    return normalized == "yes"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, required=True, help="blind_review_v1 directory")
    parser.add_argument("--expert-reviewer", action="append", default=[], help="Reviewer directory name; may be repeated")
    parser.add_argument("--recognizable-threshold", type=int, default=4)
    parser.add_argument("--completeness-threshold", type=int, default=4)
    parser.add_argument("--artifact-severe-threshold", type=int, default=2)
    parser.add_argument("--min-reviewers", type=int, default=3)
    args = parser.parse_args()

    review_dir = args.review_dir
    key_dir = review_dir / "answer_keys"
    reviewer_dirs = sorted(path for path in review_dir.glob("reviewer_*") if path.is_dir())
    if len(reviewer_dirs) < 1:
        raise FileNotFoundError(f"No reviewer_* directories in {review_dir}")

    # Observations are indexed by the hidden source identity, not randomized image IDs.
    observations: dict[tuple[str, str], list[dict]] = defaultdict(list)
    per_reviewer: dict[str, dict[tuple[str, str], dict]] = {}
    expected_identities: set[tuple[str, str]] | None = None

    for reviewer_dir in reviewer_dirs:
        reviewer = reviewer_dir.name
        score_path = reviewer_dir / "scoring_sheet.csv"
        key_path = key_dir / f"{reviewer}_key.csv"
        if not score_path.is_file() or not key_path.is_file():
            raise FileNotFoundError(f"Missing scoring sheet or answer key for {reviewer}")

        key_rows = read_csv(key_path)
        key_by_image: dict[str, tuple[str, str]] = {}
        for row in key_rows:
            image_id = row["image_id"]
            identity = (row["source"], row["source_id"])
            if image_id in key_by_image or identity in key_by_image.values():
                raise ValueError(f"Duplicate image ID or source identity in {key_path}: {image_id}/{identity}")
            key_by_image[image_id] = identity

        scores = read_csv(score_path)
        score_by_id = {row.get("image_id", ""): row for row in scores}
        if len(score_by_id) != len(scores) or set(score_by_id) != set(key_by_image):
            raise ValueError(f"Scoring IDs for {reviewer} do not exactly match its answer key")

        current: dict[tuple[str, str], dict] = {}
        for image_id, identity in key_by_image.items():
            row = score_by_id[image_id]
            if any(not row.get(field, "").strip() for field in RATING_FIELDS):
                raise ValueError(f"{reviewer}/{image_id}: incomplete rating; all 1–5 fields are required")
            parsed = {field: rating(row[field], field, reviewer, image_id) for field in RATING_FIELDS}
            parsed["repeated_template"] = yes_no(row.get("repeated_template_yes_no", ""), reviewer, image_id)
            parsed["notes"] = row.get("notes", "")
            current[identity] = parsed
            observations[identity].append(parsed)

        identities = set(current)
        if expected_identities is None:
            expected_identities = identities
        elif identities != expected_identities:
            raise ValueError(f"Reviewer item sets differ for {reviewer}; packets must cover the same blinded images")
        per_reviewer[reviewer] = current

    reviewer_count = len(reviewer_dirs)
    vote_threshold = math.ceil(2 * reviewer_count / 3)
    source_results = {}
    generated_rows = []
    for identity, ratings in sorted(observations.items()):
        source, source_id = identity
        if len(ratings) != reviewer_count:
            raise ValueError(f"{identity} has {len(ratings)} ratings; expected {reviewer_count}")
        row = {
            "source": source,
            "source_id": source_id,
            "recognizable_votes": sum(r["wbc_recognizability_1to5"] >= args.recognizable_threshold for r in ratings),
            "complete_votes": sum(r["single_cell_completeness_1to5"] >= args.completeness_threshold for r in ratings),
            "severe_artifact_votes": sum(r["artifact_severity_1to5"] <= args.artifact_severe_threshold for r in ratings),
            "repeated_template_votes": sum(r["repeated_template"] for r in ratings),
            "reviewer_count": reviewer_count,
        }
        row["recognizable_majority"] = row["recognizable_votes"] >= vote_threshold
        row["complete_majority"] = row["complete_votes"] >= vote_threshold
        row["severe_artifact_majority"] = row["severe_artifact_votes"] >= vote_threshold
        row["repeated_template_majority"] = row["repeated_template_votes"] >= vote_threshold
        source_results.setdefault(source, []).append(row)
        if source == "generated":
            generated_rows.append(row)

    if not generated_rows:
        raise ValueError("No generated images found in answer keys")

    def proportion(rows: list[dict], field: str) -> float:
        return sum(bool(row[field]) for row in rows) / len(rows)

    generated_summary = {
        "generated_count": len(generated_rows),
        "recognizable_by_at_least_two_thirds": proportion(generated_rows, "recognizable_majority"),
        "complete_by_at_least_two_thirds": proportion(generated_rows, "complete_majority"),
        "severe_artifact_by_at_least_two_thirds": proportion(generated_rows, "severe_artifact_majority"),
        "repeated_template_by_at_least_two_thirds": proportion(generated_rows, "repeated_template_majority"),
    }
    gates = {
        "at_least_minimum_reviewers": reviewer_count >= args.min_reviewers,
        "recognizable_at_least_80_percent": generated_summary["recognizable_by_at_least_two_thirds"] >= 0.80,
        "complete_at_least_80_percent": generated_summary["complete_by_at_least_two_thirds"] >= 0.80,
        "severe_artifact_at_most_15_percent": generated_summary["severe_artifact_by_at_least_two_thirds"] <= 0.15,
    }

    reviewer_rates = {}
    for reviewer, items in per_reviewer.items():
        generated_scores = [items[("generated", row["source_id"])] for row in generated_rows]
        reviewer_rates[reviewer] = {
            "recognizable_rate": sum(r["wbc_recognizability_1to5"] >= args.recognizable_threshold for r in generated_scores) / len(generated_scores),
            "complete_rate": sum(r["single_cell_completeness_1to5"] >= args.completeness_threshold for r in generated_scores) / len(generated_scores),
            "severe_artifact_rate": sum(r["artifact_severity_1to5"] <= args.artifact_severe_threshold for r in generated_scores) / len(generated_scores),
            "repeated_template_rate": sum(r["repeated_template"] for r in generated_scores) / len(generated_scores),
        }

    expert_names = set(args.expert_reviewer)
    unknown_experts = sorted(expert_names - {path.name for path in reviewer_dirs})
    if unknown_experts:
        raise ValueError(f"Unknown --expert-reviewer values: {unknown_experts}")
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "review_dir": str(review_dir),
        "reviewers": [path.name for path in reviewer_dirs],
        "reviewer_count": reviewer_count,
        "expert_reviewers_declared": sorted(expert_names),
        "expertise_limitation": None if expert_names else "No reviewer expertise was declared; interpret conclusions cautiously.",
        "thresholds": {
            "vote_rule": f"at least {vote_threshold}/{reviewer_count} reviewers",
            "recognizable_score_min": args.recognizable_threshold,
            "complete_score_min": args.completeness_threshold,
            "artifact_severe_score_max": args.artifact_severe_threshold,
            "recognizable_rate_min": 0.80,
            "complete_rate_min": 0.80,
            "severe_artifact_rate_max": 0.15,
        },
        "generated_summary": generated_summary,
        "source_counts": {source: len(rows) for source, rows in source_results.items()},
        "gates": gates,
        "quality_gate_result": "pass" if all(gates.values()) else "no_go_or_insufficient_review",
        "reviewer_level_generated_rates": reviewer_rates,
        "limitations": [
            "This is a project-specific image-quality gate, not clinical validation.",
            "Reviewer expertise is self-declared and must be documented independently.",
            "Repeated-template ratings are a subjective flag and must be interpreted with feature-coverage and memorization results.",
        ],
    }
    out_json = review_dir / "blind_review_summary.json"
    out_md = review_dir / "blind_review_summary.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = generated_summary
    lines = [
        "# Blinded WBC Review Summary",
        "",
        f"- Reviewers: {reviewer_count}",
        f"- Generated images: {summary['generated_count']}",
        f"- Vote rule: at least {vote_threshold}/{reviewer_count}",
        f"- Recognizable WBC: {summary['recognizable_by_at_least_two_thirds']:.1%} (gate ≥80%)",
        f"- Complete single-cell image: {summary['complete_by_at_least_two_thirds']:.1%} (gate ≥80%)",
        f"- Severe artifacts: {summary['severe_artifact_by_at_least_two_thirds']:.1%} (gate ≤15%)",
        f"- Repeated-template flags: {summary['repeated_template_by_at_least_two_thirds']:.1%} (descriptive)",
        f"- Quality gate: **{payload['quality_gate_result']}**",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in gates.items())
    lines.extend(["", "## Per-reviewer rates", ""])
    lines.extend(
        f"- {reviewer}: recognizable {rates['recognizable_rate']:.1%}; complete {rates['complete_rate']:.1%}; severe artifacts {rates['severe_artifact_rate']:.1%}; repeated-template {rates['repeated_template_rate']:.1%}"
        for reviewer, rates in reviewer_rates.items()
    )
    lines.extend(["", "See `blind_review_summary.json` for thresholds, limitations, and machine-readable results.", ""])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(out_json), "markdown": str(out_md), "quality_gate_result": payload["quality_gate_result"]}, indent=2))


if __name__ == "__main__":
    main()
