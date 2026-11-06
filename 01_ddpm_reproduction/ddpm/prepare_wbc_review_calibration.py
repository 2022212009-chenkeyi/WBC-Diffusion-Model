"""Prepare a reproducible, real-train-image set for reviewer scale calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import tomllib
from pathlib import Path

from PIL import Image, ImageDraw


SCORE_FIELDS = (
    "image_id",
    "wbc_recognizability_1to5",
    "single_cell_completeness_1to5",
    "nucleus_cytoplasm_structure_1to5",
    "stain_background_1to5",
    "artifact_severity_1to5",
    "repeated_template_yes_no",
    "notes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--output-dir", type=Path, default=Path("../results/phase2_dataset/wbc_review_calibration_v1"))
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    image_size = int(config["model"]["image_size"])
    manifest_path = Path(data_cfg["split_manifest"])
    image_root = Path(data_cfg["image_root"])
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["split"] == "train" and row["qc_decision"] == "keep"
        ]
    if args.count > len(rows):
        raise ValueError(f"Requested {args.count} examples, only {len(rows)} train/keep images exist")

    selected = random.Random(args.seed).sample(rows, args.count)
    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    thumb_size = 192
    columns = 5
    rows_per_page = (args.count + columns - 1) // columns
    page = Image.new("RGB", (columns * (thumb_size + 16), rows_per_page * (thumb_size + 42)), "white")
    draw = ImageDraw.Draw(page)
    score_path = output_dir / "calibration_scoring_sheet.csv"

    with score_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        for index, row in enumerate(selected, start=1):
            image_id = f"CAL-{index:03d}"
            source_path = image_root / row["relative_path"]
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            with Image.open(source_path) as source:
                image = source.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
            image.save(image_dir / f"{image_id}.png")
            x = ((index - 1) % columns) * (thumb_size + 16)
            y = ((index - 1) // columns) * (thumb_size + 42)
            page.paste(image.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS), (x, y + 22))
            draw.text((x + 2, y + 4), image_id, fill="black")
            writer.writerow({"image_id": image_id})
            manifest_rows.append(
                {
                    "image_id": image_id,
                    "source": "real_train",
                    "relative_path": row["relative_path"],
                    "source_sha256": sha256_file(source_path),
                    "resized_image_sha256": sha256_file(image_dir / f"{image_id}.png"),
                }
            )

    page_path = output_dir / "calibration_contact_sheet.jpg"
    page.save(page_path, quality=92)
    with (output_dir / "calibration_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_dir / "README.md").write_text(
        "# WBC Reviewer Calibration Set\n\n"
        f"This reproducible set contains {args.count} QC-kept real training images, selected with seed `{args.seed}` from the frozen training split and resized using the baseline RGB/Lanczos preprocessing ({image_size}×{image_size}).\n\n"
        "Use these labeled real examples to align how reviewers interpret the scoring scale before they receive any generated images. Discuss score meanings only; do not change the preregistered pass thresholds based on this set. These examples are from the training split and are not part of the 200-image held-out test review.\n\n"
        "Artifact severity is reverse-coded: 1=severe/unusable, 3=visible but not severe, 5=no obvious artifact. The other 1–5 scales use 1=poor/unrecognizable and 5=clear/complete.\n\n"
        "`calibration_manifest.csv` records source paths and hashes. `calibration_scoring_sheet.csv` is optional for individual practice; it is not part of the formal blind-review analysis.\n",
        encoding="utf-8",
    )
    print(f"calibration_set={output_dir}")
    print(f"contact_sheet={page_path}")
    print(f"images={args.count}")


if __name__ == "__main__":
    main()
