"""Exact-pixel and 64-bit dHash nearest-neighbor screen against WBC train images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np
from PIL import Image, ImageDraw


def resize_rgb(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def dhash(image: Image.Image) -> int:
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def sha256_pixels(image: Image.Image) -> str:
    return hashlib.sha256(np.asarray(image, dtype=np.uint8).tobytes()).hexdigest()


def write_contact_pages(candidates: list[dict], train_by_relpath: dict[str, Path], generated: np.ndarray, out_dir: Path) -> list[str]:
    page_paths: list[str] = []
    thumb = 112
    rows_per_page = 12
    for page_start in range(0, len(candidates), rows_per_page):
        page = candidates[page_start : page_start + rows_per_page]
        canvas = Image.new("RGB", (thumb * 2 + 16, (thumb + 24) * len(page) + 10), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, record in enumerate(page):
            top = row_index * (thumb + 24)
            gen = Image.fromarray(np.array(generated[int(record["generated_index"])], copy=True), mode="RGB")
            train_path = train_by_relpath[record["nearest_train_relative_path"]]
            with Image.open(train_path) as source:
                ref = resize_rgb(source, thumb)
            canvas.paste(gen.resize((thumb, thumb), Image.Resampling.LANCZOS), (0, top + 20))
            canvas.paste(ref, (thumb + 16, top + 20))
            draw.text((2, top + 2), f"gen {record['generated_index']} / dHash {record['nearest_dhash_hamming']}", fill="black")
        path = out_dir / f"nearest_neighbor_review_{len(page_paths)+1:03d}.jpg"
        canvas.save(path, quality=92)
        page_paths.append(str(path))
    return page_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True, help="Completed uint8 NHWC .npy from ddpm.sample")
    parser.add_argument("--threshold", type=int, default=4)
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text())
    exp, data_cfg, model_cfg = config["experiment"], config["data"], config["model"]
    samples = np.load(args.samples, mmap_mode="r")
    size = int(model_cfg["image_size"])
    if samples.dtype != np.uint8 or samples.ndim != 4 or samples.shape[1:] != (size, size, model_cfg["channels"]):
        raise ValueError(f"expected uint8 NHWC (*,{size},{size},{model_cfg['channels']}), got {samples.dtype} {samples.shape}")

    manifest_path = Path(data_cfg["split_manifest"])
    image_root = Path(data_cfg["image_root"])
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        train_rows = [
            row for row in csv.DictReader(stream)
            if row["split"] == "train" and row["qc_decision"] == "keep"
        ]
    train_hashes: dict[str, list[str]] = {}
    train_items: list[tuple[str, int]] = []
    train_by_relpath: dict[str, Path] = {}
    for index, row in enumerate(train_rows):
        path = image_root / row["relative_path"]
        with Image.open(path) as source:
            resized = resize_rgb(source, size)
        exact_hash = sha256_pixels(resized)
        train_hashes.setdefault(exact_hash, []).append(row["relative_path"])
        train_items.append((row["relative_path"], dhash(resized)))
        train_by_relpath[row["relative_path"]] = path
        if (index + 1) % 2000 == 0:
            print(f"hashed_train={index+1}/{len(train_rows)}", flush=True)

    run_suffix = exp.get("run_suffix", "")
    run_name = f"{exp['name']}-{run_suffix}" if run_suffix else exp["name"]
    output_root = Path(exp["output_dir"])
    output_dir = output_root / "results" / run_name / "evaluation" / "memorization"
    output_dir.mkdir(parents=True, exist_ok=True)

    nearest_path = output_dir / "generated_nearest_train.csv"
    candidate_path = output_dir / "dhash_candidates_le_threshold.csv"
    candidate_rows: list[dict] = []
    exact_count = 0
    with nearest_path.open("w", newline="", encoding="utf-8") as nearest_stream:
        nearest_writer = csv.DictWriter(
            nearest_stream,
            fieldnames=("generated_index", "exact_pixel_match", "exact_train_relative_paths", "nearest_train_relative_path", "nearest_dhash_hamming"),
        )
        nearest_writer.writeheader()
        for generated_index in range(len(samples)):
            generated_image = Image.fromarray(np.array(samples[generated_index], copy=True), mode="RGB")
            exact_hash = sha256_pixels(generated_image)
            exact_paths = train_hashes.get(exact_hash, [])
            exact_count += bool(exact_paths)
            generated_hash = dhash(generated_image)
            nearest_relpath, nearest_hash = min(
                train_items,
                key=lambda item: (generated_hash ^ item[1]).bit_count(),
            )
            nearest_distance = (generated_hash ^ nearest_hash).bit_count()
            record = {
                "generated_index": generated_index,
                "exact_pixel_match": bool(exact_paths),
                "exact_train_relative_paths": ";".join(exact_paths),
                "nearest_train_relative_path": nearest_relpath,
                "nearest_dhash_hamming": nearest_distance,
            }
            nearest_writer.writerow(record)
            if record["nearest_dhash_hamming"] <= args.threshold:
                candidate_rows.append(record)
            if (generated_index + 1) % 100 == 0 or generated_index + 1 == len(samples):
                print(f"matched_generated={generated_index+1}/{len(samples)}", flush=True)

    with candidate_path.open("w", newline="", encoding="utf-8") as candidate_stream:
        writer = csv.DictWriter(candidate_stream, fieldnames=("generated_index", "exact_pixel_match", "exact_train_relative_paths", "nearest_train_relative_path", "nearest_dhash_hamming"))
        writer.writeheader()
        writer.writerows(candidate_rows)
    contact_pages = write_contact_pages(candidate_rows, train_by_relpath, samples, output_dir)
    sample_metadata_path = args.samples.with_suffix(".json")
    sample_metadata = json.loads(sample_metadata_path.read_text()) if sample_metadata_path.is_file() else {}
    report = {
        "run_name": run_name,
        "generated_samples_file": str(args.samples),
        "checkpoint_step": sample_metadata.get("checkpoint_step"),
        "generated_count": int(len(samples)),
        "train_count": len(train_items),
        "preprocessing": f"RGB full-frame resize to {size}x{size} with PIL Lanczos; exact pixel SHA-256 and 64-bit dHash (9x8 grayscale)",
        "dhash_review_threshold": args.threshold,
        "exact_pixel_match_count": int(exact_count),
        "dhash_candidate_count": len(candidate_rows),
        "dhash_candidates_are_not_confirmed_copies": True,
        "nearest_neighbor_csv": str(nearest_path),
        "candidate_csv": str(candidate_path),
        "visual_review_pages": contact_pages,
        "limitations": [
            "dHash candidates require visual review and can include distinct lookalike cells.",
            "Near-duplicate hashing can miss translated, resized, recolored, or otherwise transformed copies.",
            "Only the frozen training split is searched because only it was visible to the model.",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
