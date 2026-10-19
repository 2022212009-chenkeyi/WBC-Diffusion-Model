#!/usr/bin/env python3
"""Verify and safely extract a ZIP to a new directory with an expanded-size cap."""

from __future__ import annotations

import argparse
import json
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath


def safe_member(info: zipfile.ZipInfo) -> tuple[PurePosixPath, bool]:
    name = info.filename
    if not name or "\\" in name or "\0" in name:
        raise ValueError(f"Invalid ZIP member path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    is_dir = info.is_dir()
    if kind == stat.S_IFLNK or kind not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"Links/special files are not accepted: {name!r}")
    if is_dir != (kind == stat.S_IFDIR) and kind != 0:
        raise ValueError(f"Inconsistent ZIP member type: {name!r}")
    if not is_dir and kind == stat.S_IFDIR:
        raise ValueError(f"Directory has file name: {name!r}")
    return path, is_dir


def inspect_and_extract(archive: Path, destination: Path, cap: int) -> dict:
    with zipfile.ZipFile(archive) as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"CRC check failed: {bad_member}")

        members = []
        seen: set[str] = set()
        expanded = 0
        files = 0
        for info in zf.infolist():
            rel, is_dir = safe_member(info)
            key = rel.as_posix().rstrip("/")
            if key in seen:
                raise ValueError(f"Duplicate ZIP path: {key!r}")
            seen.add(key)
            if not is_dir:
                if info.file_size < 0:
                    raise ValueError(f"Invalid member size: {key!r}")
                expanded += info.file_size
                files += 1
                if expanded > cap:
                    raise ValueError(f"Expanded ZIP exceeds cap ({cap} bytes)")
            members.append((info, rel, is_dir))

        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir()
        root = destination
        try:
            for info, rel, is_dir in members:
                target = root.joinpath(*rel.parts)
                if is_dir:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with zf.open(info, "r") as src, target.open("xb") as dst:
                    while chunk := src.read(1024 * 1024):
                        written += len(chunk)
                        if written > info.file_size:
                            raise ValueError(f"Member exceeds declared size: {info.filename!r}")
                        dst.write(chunk)
                if written != info.file_size:
                    raise ValueError(f"Member size mismatch: {info.filename!r}")
        except Exception:
            raise

    return {"archive": str(archive), "archive_bytes": archive.stat().st_size,
            "entry_count": len(members), "file_count": files,
            "expanded_file_bytes": expanded, "extracted_to": str(destination)}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-total-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.max_total_bytes <= 0:
        parser.error("--max-total-bytes must be positive")
    print(json.dumps(inspect_and_extract(args.archive, args.destination,
                                         args.max_total_bytes), indent=2))


if __name__ == "__main__":
    main()
