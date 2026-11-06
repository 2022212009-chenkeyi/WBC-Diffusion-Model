"""Create blinded image packets and empty scoring sheets for WBC image review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path
import tomllib

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


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
    parser.add_argument("--samples", type=Path, required=True, help="Completed uint8 NHWC .npy from ddpm.sample")
    parser.add_argument("--count-per-source", type=int, default=200)
    parser.add_argument("--reviewers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.count_per_source < 1 or args.reviewers < 1:
        raise ValueError("count-per-source and reviewers must be positive")

    config = tomllib.loads(args.config.read_text())
    exp, data_cfg, model_cfg = config["experiment"], config["data"], config["model"]
    samples = np.load(args.samples, mmap_mode="r")
    count = args.count_per_source
    size = int(model_cfg["image_size"])
    if samples.dtype != np.uint8 or samples.ndim != 4 or samples.shape[1:] != (size, size, model_cfg["channels"]):
        raise ValueError(f"expected uint8 NHWC (*,{size},{size},{model_cfg['channels']}), got {samples.dtype} {samples.shape}")
    if len(samples) < count:
        raise ValueError(f"need at least {count} generated samples, found {len(samples)}")

    manifest_path = Path(data_cfg["split_manifest"])
    image_root = Path(data_cfg["image_root"])
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        real_relpaths = [
            row["relative_path"] for row in csv.DictReader(stream)
            if row["split"] == "test" and row["qc_decision"] == "keep"
        ]
    if len(real_relpaths) < count:
        raise ValueError(f"need {count} QC-kept test images, found {len(real_relpaths)}")
    selection_rng = random.Random(args.seed)
    generated_indices = selection_rng.sample(range(len(samples)), count)
    real_relpaths_selected = selection_rng.sample(real_relpaths, count)
    resize = transforms.Resize((size, size), interpolation=InterpolationMode.LANCZOS)

    run_suffix = exp.get("run_suffix", "")
    run_name = f"{exp['name']}-{run_suffix}" if run_suffix else exp["name"]
    output_root = Path(exp["output_dir"])
    output_dir = args.output_dir or output_root / "results" / run_name / "evaluation" / "blind_review_v1"
    answer_dir = output_dir / "answer_keys"
    answer_dir.mkdir(parents=True, exist_ok=True)

    for reviewer_index in range(1, args.reviewers + 1):
        reviewer_id = f"reviewer_{reviewer_index:02d}"
        reviewer_dir = output_dir / reviewer_id
        image_dir = reviewer_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        items = [("generated", index, None) for index in generated_indices]
        items.extend(("real_test", None, relative_path) for relative_path in real_relpaths_selected)
        random.Random(args.seed + reviewer_index).shuffle(items)
        key_rows = []
        with (reviewer_dir / "scoring_sheet.csv").open("w", newline="", encoding="utf-8") as score_stream:
            score_writer = csv.DictWriter(score_stream, fieldnames=SCORE_FIELDS)
            score_writer.writeheader()
            for position, (source, generated_index, relative_path) in enumerate(items, start=1):
                image_id = f"R{reviewer_index:02d}_{position:04d}"
                if source == "generated":
                    image = Image.fromarray(np.array(samples[generated_index], copy=True), mode="RGB")
                    source_id = str(generated_index)
                else:
                    source_path = image_root / relative_path
                    with Image.open(source_path) as opened:
                        image = resize(opened.convert("RGB"))
                    source_id = relative_path
                image.save(image_dir / f"{image_id}.png")
                score_writer.writerow({"image_id": image_id})
                key_rows.append({"image_id": image_id, "source": source, "source_id": source_id})
        with (answer_dir / f"{reviewer_id}_key.csv").open("w", newline="", encoding="utf-8") as key_stream:
            writer = csv.DictWriter(key_stream, fieldnames=("image_id", "source", "source_id"))
            writer.writeheader()
            writer.writerows(key_rows)

    (output_dir / "review_instructions_zh.md").write_text(
        "# WBC 图像盲评说明\n\n"
        "每位评审打开自己目录中的 `images/`，按文件名顺序评分，并填写 `scoring_sheet.csv`。\n"
        "请在完成评分前不要查看 `answer_keys/`。图像已随机混排，不显示来源。\n\n"
        "可辨认、单细胞完整性、核/胞质结构、染色/背景四项均按 1–5 分：1=极差/无法判断，3=部分可辨但有明显问题，5=清楚且合理。\n"
        "伪影严重程度按反向 1–5 分：1=严重噪声、明显结构破坏、严重截断/多细胞、空白或其他使图像不可用的问题；3=可见但不严重；5=无明显伪影。协议将 ≤2 计为严重伪影。\n"
        "重复模板标记：与本包其他图像明显只是同一外观的微小变化；不要仅因同一细胞类别就标记。\n"
        "只评价图像外观，不判断疾病、诊断或病理机制。\n"
    )
    (output_dir / "packet_manifest.txt").write_text(
        f"run_name={run_name}\nsamples_file={args.samples}\nsamples_sha256={sha256_file(args.samples)}\n"
        f"split_manifest={manifest_path}\nselection_seed={args.seed}\ncount_per_source={count}\nreviewers={args.reviewers}\n"
        f"image_size={size}\nreal_source_split=test\n"
    )
    print(f"blind review packets ready: {output_dir}")
    print(f"Keep answer keys private until all scoring sheets have been returned: {answer_dir}")


if __name__ == "__main__":
    main()
