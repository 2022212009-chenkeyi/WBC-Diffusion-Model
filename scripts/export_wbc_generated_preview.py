#!/usr/bin/env python3
"""Export a reproducible human-viewable subset of final DDPM samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("/projects/keyi005ssd/WBCDiffusion/generated_samples/wbc-raabin-256-baseline-v1/samples_ema_step0200000_n2000_seed2026.npy"),
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=20261106)
    parser.add_argument("--output-dir", type=Path, default=Path("results/wbc_unconditional_baseline_001/generated_preview_100"))
    parser.add_argument("--columns", type=int, default=10)
    args = parser.parse_args()
    samples = np.load(args.samples, mmap_mode="r")
    if samples.dtype != np.uint8 or samples.ndim != 4 or samples.shape[1:] != (256, 256, 3):
        raise ValueError(f"expected uint8 NHWC (*,256,256,3), got {samples.dtype} {samples.shape}")
    if not 1 <= args.count <= len(samples) or args.columns < 1:
        raise ValueError("count must be within the sample count and columns must be positive")

    rng = np.random.default_rng(args.selection_seed)
    selected = sorted(int(i) for i in rng.choice(len(samples), size=args.count, replace=False))
    out_dir = args.output_dir.resolve()
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    thumb, caption, gap = 144, 23, 6
    rows = (args.count + args.columns - 1) // args.columns
    tile_w, tile_h = thumb + gap, thumb + caption + gap
    sheet = Image.new("RGB", (args.columns * tile_w + gap, rows * tile_h + gap), "white")
    draw = ImageDraw.Draw(sheet)
    manifest = []

    for preview_number, sample_index in enumerate(selected, start=1):
        image = Image.fromarray(np.array(samples[sample_index], copy=True), mode="RGB")
        filename = f"generated_{preview_number:03d}_sample_{sample_index:04d}.png"
        image.save(image_dir / filename, optimize=True)
        x = gap + ((preview_number - 1) % args.columns) * tile_w
        y = gap + ((preview_number - 1) // args.columns) * tile_h
        sheet.paste(image.resize((thumb, thumb), Image.Resampling.LANCZOS), (x, y + caption))
        draw.text((x + 2, y + 3), f"#{preview_number:03d} / idx {sample_index}", fill="#202a38")
        manifest.append({"preview_number": preview_number, "sample_index": sample_index, "filename": filename})

    sheet_path = out_dir / "contact_sheet.jpg"
    sheet.save(sheet_path, quality=94, optimize=True)
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("preview_number", "sample_index", "filename"))
        writer.writeheader()
        writer.writerows(manifest)
    metadata = {
        "source_samples": str(args.samples),
        "checkpoint_step": 200000,
        "sample_seed": 2026,
        "preview_count": args.count,
        "selection_seed": args.selection_seed,
        "selection_without_replacement": True,
        "image_size": "256x256 RGB",
        "note": "Generated-only preview, separate from blinded test review. Viewing this does not constitute reviewer scoring.",
    }
    (out_dir / "README.md").write_text(
        "# DDPM 生成样本预览\n\n"
        f"本目录从最终 EMA checkpoint 的 2,000 张生成样本中固定抽取 {args.count} 张，供直接浏览。打开 `contact_sheet.jpg` 查看总览，或进入 `images/` 查看单张原图。`manifest.csv` 记录了预览编号到原始样本索引的对应关系。\n\n"
        "这是生成图单独预览，不是盲评结果；它不包含真实 test 对照。正式质量门槛仍以独立评审填写的盲评表为准。\n\n"
        f"选择 seed：`{args.selection_seed}`。样本来源和参数见 `metadata.json`。\n",
        encoding="utf-8",
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"preview_dir={out_dir}")
    print(f"contact_sheet={sheet_path}")
    print(f"exported_images={len(selected)}")


if __name__ == "__main__":
    main()
