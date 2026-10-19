#!/usr/bin/env python3
"""Create a read-only inventory of a dataset directory.

This script records file paths, sizes, extensions, immediate-parent counts,
and (when Pillow is installed) image dimensions/read errors. It does not infer
that a directory name is a semantic class label and never modifies the input.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError:
        return {"inspection": "Pillow is not installed"}

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
            }
    except Exception as exc:  # Pillow exposes several decoder-specific errors.
        return {"error": f"{type(exc).__name__}: {exc}"}


def inventory(root: Path, hash_files: bool = False) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Input is not a directory: {root}")

    files: list[dict[str, Any]] = []
    extensions: collections.Counter[str] = collections.Counter()
    immediate_parent_counts: collections.Counter[str] = collections.Counter()
    image_count = 0
    image_errors = 0
    exact_hashes: dict[str, list[str]] = collections.defaultdict(list)

    for current, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            path = Path(current, filename)
            relative = path.relative_to(root).as_posix()
            suffix = path.suffix.lower()
            is_image = suffix in IMAGE_EXTENSIONS
            try:
                stat = path.stat(follow_symlinks=False)
            except OSError as exc:
                files.append({"path": relative, "error": f"{type(exc).__name__}: {exc}"})
                continue

            row: dict[str, Any] = {
                "path": relative,
                "size_bytes": stat.st_size,
                "extension": suffix or "[none]",
                "is_image_by_extension": is_image,
            }
            extensions[suffix or "[none]"] += 1
            parent = Path(relative).parent.as_posix()
            immediate_parent = "." if parent == "." else parent.split("/")[0]
            immediate_parent_counts[immediate_parent] += 1

            if is_image:
                image_count += 1
                row["image_inspection"] = inspect_image(path)
                if "error" in row["image_inspection"]:
                    image_errors += 1

            if hash_files:
                try:
                    file_hash = sha256_file(path)
                    row["sha256"] = file_hash
                    exact_hashes[file_hash].append(relative)
                except OSError as exc:
                    row["hash_error"] = f"{type(exc).__name__}: {exc}"
            files.append(row)

    duplicate_groups = [
        {"sha256": digest, "paths": paths}
        for digest, paths in sorted(exact_hashes.items())
        if len(paths) > 1
    ]
    return {
        "root": str(root),
        "note": (
            "Immediate-parent counts describe directory structure only; they "
            "are not asserted to be semantic class counts."
        ),
        "summary": {
            "file_count": len(files),
            "image_count_by_extension": image_count,
            "image_decode_errors": image_errors,
            "extensions": dict(sorted(extensions.items())),
            "files_by_top_level_directory": dict(sorted(immediate_parent_counts.items())),
            "exact_duplicate_groups": len(duplicate_groups),
            "hashing_enabled": hash_files,
        },
        "exact_duplicate_groups": duplicate_groups,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Dataset directory to inspect (read-only)")
    parser.add_argument("--output", type=Path, help="Write JSON report here; default is stdout")
    parser.add_argument(
        "--hash-files",
        action="store_true",
        help="Calculate SHA-256 for every file to identify exact duplicates (can be slow)",
    )
    args = parser.parse_args()
    report = inventory(args.root, hash_files=args.hash_files)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
