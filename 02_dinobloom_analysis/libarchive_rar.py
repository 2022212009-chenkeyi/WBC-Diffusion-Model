#!/usr/bin/env python3
"""List or safely extract RAR archives using the system libarchive library.

The extraction mode refuses absolute/traversal paths, links, special files,
duplicate paths, an existing destination, and archives exceeding a byte cap.
It only writes into a newly created destination directory.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

ARCHIVE_OK = 0
ARCHIVE_EOF = 1
AE_IFREG = stat.S_IFREG
AE_IFDIR = stat.S_IFDIR
CHUNK = 1024 * 1024


class LibArchive:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libarchive.so.13")
        self.lib.archive_read_new.restype = ctypes.c_void_p
        self.lib.archive_read_support_filter_all.argtypes = [ctypes.c_void_p]
        self.lib.archive_read_support_format_rar.argtypes = [ctypes.c_void_p]
        self.lib.archive_read_support_format_rar5.argtypes = [ctypes.c_void_p]
        self.lib.archive_read_open_filename.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.archive_read_next_header.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.archive_read_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.archive_read_data.restype = ctypes.c_ssize_t
        self.lib.archive_error_string.argtypes = [ctypes.c_void_p]
        self.lib.archive_error_string.restype = ctypes.c_char_p
        self.lib.archive_entry_pathname_utf8.argtypes = [ctypes.c_void_p]
        self.lib.archive_entry_pathname_utf8.restype = ctypes.c_char_p
        self.lib.archive_entry_size.argtypes = [ctypes.c_void_p]
        self.lib.archive_entry_size.restype = ctypes.c_int64
        self.lib.archive_entry_filetype.argtypes = [ctypes.c_void_p]
        self.lib.archive_entry_filetype.restype = ctypes.c_uint
        self.lib.archive_entry_symlink.argtypes = [ctypes.c_void_p]
        self.lib.archive_entry_symlink.restype = ctypes.c_char_p
        self.lib.archive_entry_hardlink.argtypes = [ctypes.c_void_p]
        self.lib.archive_entry_hardlink.restype = ctypes.c_char_p
        self.lib.archive_read_free.argtypes = [ctypes.c_void_p]
        self.lib.archive_read_close.argtypes = [ctypes.c_void_p]

    def error(self, reader: int) -> str:
        message = self.lib.archive_error_string(reader)
        return message.decode("utf-8", "replace") if message else "unknown libarchive error"


def safe_relative_path(raw: bytes | None) -> tuple[str, PurePosixPath]:
    if not raw:
        raise ValueError("Archive member has an empty pathname")
    name = raw.decode("utf-8", "strict")
    path = PurePosixPath(name)
    if "\\" in name or path.is_absolute() or any(part in ("..", "") for part in path.parts):
        raise ValueError(f"Unsafe archive pathname: {name!r}")
    if not path.parts or path.parts[0].endswith(":"):
        raise ValueError(f"Unsafe archive pathname: {name!r}")
    return name, path


def inspect(archive_path: Path, destination: Path | None, max_total_bytes: int | None) -> dict[str, Any]:
    archive_path = archive_path.resolve(strict=True)
    binding = LibArchive()
    lib = binding.lib
    reader = lib.archive_read_new()
    if not reader:
        raise RuntimeError("libarchive could not allocate a reader")
    root: Path | None = None
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_bytes = 0
    file_count = 0
    dir_count = 0
    try:
        for configure in (lib.archive_read_support_filter_all,
                          lib.archive_read_support_format_rar,
                          lib.archive_read_support_format_rar5):
            status = configure(reader)
            if status < ARCHIVE_OK:
                raise RuntimeError(binding.error(reader))
        status = lib.archive_read_open_filename(reader, os.fsencode(archive_path), 10240)
        if status != ARCHIVE_OK:
            raise RuntimeError(binding.error(reader))

        entry = ctypes.c_void_p()
        while True:
            status = lib.archive_read_next_header(reader, ctypes.byref(entry))
            if status == ARCHIVE_EOF:
                break
            if status != ARCHIVE_OK:
                raise RuntimeError(binding.error(reader))
            raw_name = lib.archive_entry_pathname_utf8(entry)
            name, relative = safe_relative_path(raw_name)
            if name in seen:
                raise ValueError(f"Duplicate archive pathname: {name!r}")
            seen.add(name)
            filetype = lib.archive_entry_filetype(entry)
            size = int(lib.archive_entry_size(entry))
            if size < 0:
                raise ValueError(f"Unknown/negative member size for {name!r}")
            if lib.archive_entry_symlink(entry) or lib.archive_entry_hardlink(entry):
                raise ValueError(f"Links are not accepted in dataset archives: {name!r}")
            if filetype not in (AE_IFREG, AE_IFDIR):
                raise ValueError(f"Special file type is not accepted: {name!r} ({filetype})")
            entry_info = {"path": name, "type": "directory" if filetype == AE_IFDIR else "file",
                          "size_bytes": 0 if filetype == AE_IFDIR else size}
            entries.append(entry_info)

            if filetype == AE_IFDIR:
                dir_count += 1
                if destination is not None:
                    if root is None:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.mkdir()
                        root = destination
                    (root / Path(*relative.parts)).mkdir(parents=True, exist_ok=True)
                continue

            file_count += 1
            total_bytes += size
            if max_total_bytes is not None and total_bytes > max_total_bytes:
                raise ValueError(f"Expanded data exceeds byte cap ({max_total_bytes})")

            output = None
            if destination is not None:
                if root is None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.mkdir()
                    root = destination
                output = root.joinpath(*relative.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                output = output.open("xb")
            actual_size = 0
            try:
                if output is None:
                    while True:
                        chunk = ctypes.create_string_buffer(CHUNK)
                        amount = lib.archive_read_data(reader, chunk, CHUNK)
                        if amount == 0:
                            break
                        if amount < 0:
                            raise RuntimeError(binding.error(reader))
                        actual_size += amount
                        if actual_size > size:
                            raise ValueError(f"Member data exceeds its declared size: {name!r}")
                        if max_total_bytes is not None and total_bytes - size + actual_size > max_total_bytes:
                            raise ValueError(f"Expanded data exceeds byte cap ({max_total_bytes})")
                else:
                    with output:
                        while True:
                            chunk = ctypes.create_string_buffer(CHUNK)
                            amount = lib.archive_read_data(reader, chunk, CHUNK)
                            if amount == 0:
                                break
                            if amount < 0:
                                raise RuntimeError(binding.error(reader))
                            actual_size += amount
                            if actual_size > size:
                                raise ValueError(f"Member data exceeds its declared size: {name!r}")
                            if max_total_bytes is not None and total_bytes - size + actual_size > max_total_bytes:
                                raise ValueError(f"Expanded data exceeds byte cap ({max_total_bytes})")
                            output.write(chunk.raw[:amount])
            finally:
                if output is not None and not output.closed:
                    output.close()
            if actual_size != size:
                raise ValueError(f"Member size mismatch for {name!r}: declared {size}, read {actual_size}")
    finally:
        lib.archive_read_close(reader)
        lib.archive_read_free(reader)

    return {"archive": str(archive_path), "entry_count": len(entries),
            "file_count": file_count, "directory_count": dir_count,
            "expanded_file_bytes": total_bytes, "entries": entries,
            "extracted_to": str(root) if root else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-json", type=Path, help="Write verified member list to JSON (reads all data)")
    mode.add_argument("--extract", type=Path, help="Extract to a new directory after validating every member")
    parser.add_argument("--max-total-bytes", type=int,
                        help="Required for extraction; cap total uncompressed regular-file bytes")
    args = parser.parse_args()
    if args.extract and (args.max_total_bytes is None or args.max_total_bytes <= 0):
        parser.error("--extract requires a positive --max-total-bytes value")
    report = inspect(args.archive, args.extract,
                     args.max_total_bytes if args.extract else None)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.list_json:
        args.list_json.parent.mkdir(parents=True, exist_ok=True)
        args.list_json.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
