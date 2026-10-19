#!/usr/bin/env python3
"""Resumable, verified HTTP Range downloader for large public archives."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

CHUNK = 1024 * 1024


def fetch_range(url: str, start: int, end: int, total: int, target: Path,
                attempts: int, timeout: int) -> Path:
    expected = end - start + 1
    if target.exists() and target.stat().st_size == expected:
        return target
    temp = target.with_suffix(target.suffix + ".tmp")
    headers = target.with_suffix(target.suffix + ".headers")
    for attempt in range(1, attempts + 1):
        try:
            if temp.exists():
                temp.unlink()
            result = subprocess.run(
                ["curl", "--silent", "--show-error", "--fail", "--location",
                 "--range", f"{start}-{end}",
                 "--max-time", str(timeout), "--speed-time", str(min(120, timeout)),
                 "--speed-limit", "1024", "-D", str(headers), "-o", str(temp), url],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            received = temp.stat().st_size if temp.exists() else 0
            header_text = headers.read_text(encoding="latin-1", errors="replace") if headers.exists() else ""
            matches = re.findall(r"(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)", header_text)
            valid_header = bool(matches)
            if valid_header:
                got_start, got_end, got_total = map(int, matches[-1])
                valid_header = (got_start, got_end, got_total) == (start, end, total)
            if received == expected and valid_header:
                os.replace(temp, target)
                return target
            if received > expected:
                raise RuntimeError(f"Range {start}-{end} wrote {received} bytes; expected {expected}")
            detail = result.stderr.strip() or f"HTTP range incomplete ({received}/{expected} bytes)"
            raise RuntimeError(detail)
        except Exception as exc:
            if temp.exists():
                temp.unlink()
            if attempt == attempts:
                if expected > 64 * 1024:
                    midpoint = start + expected // 2 - 1
                    left = target.with_name(target.name + f".sub-{start}-{midpoint}.part")
                    right = target.with_name(target.name + f".sub-{midpoint + 1}-{end}.part")
                    print(f"Splitting persistently failing range {start}-{end} into smaller ranges", flush=True)
                    fetch_range(url, start, midpoint, total, left, attempts, timeout)
                    fetch_range(url, midpoint + 1, end, total, right, attempts, timeout)
                    combined = target.with_name(target.name + ".combined.tmp")
                    with combined.open("wb") as out:
                        with left.open("rb") as source:
                            shutil.copyfileobj(source, out, length=1024 * 1024)
                        with right.open("rb") as source:
                            shutil.copyfileobj(source, out, length=1024 * 1024)
                    if combined.stat().st_size != expected:
                        raise RuntimeError(f"Split ranges for {start}-{end} assembled to the wrong length")
                    os.replace(combined, target)
                    return target
                raise
            print(f"Range {start}-{end} attempt {attempt}/{attempts} incomplete: {exc}", flush=True)
            time.sleep(min(5 * attempt, 30))
    raise RuntimeError(f"Failed to fetch range {start}-{end}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("prefix", type=Path, help="Existing contiguous prefix, normally the .part archive")
    parser.add_argument("output", type=Path, help="Final complete archive path")
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=CHUNK)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    if not args.url.startswith("https://"):
        parser.error("Only HTTPS sources are accepted")
    if args.chunk_size <= 0 or args.attempts <= 0 or args.workers <= 0:
        parser.error("chunk size, workers, and attempts must be positive")
    prefix_size = args.prefix.stat().st_size
    if prefix_size < 0 or prefix_size > args.expected_size:
        raise ValueError(f"Prefix size {prefix_size} is outside expected archive size")
    if args.prefix.resolve() == args.output.resolve():
        pass
    elif args.output.exists():
        raise FileExistsError(f"Output exists and is not the prefix: {args.output}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    ranges = []
    start = prefix_size
    while start < args.expected_size:
        end = min(start + args.chunk_size - 1, args.expected_size - 1)
        chunk_path = args.work_dir / f"range-{start:012d}-{end:012d}.part"
        ranges.append((start, end, chunk_path))
        start = end + 1

    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_range, args.url, first, last, args.expected_size,
                               path, args.attempts, args.timeout)
                   for first, last, path in ranges]
        for number, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            completed.append(future.result())
            print(f"Verified HTTP ranges complete: {number}/{len(ranges)}", flush=True)
    completed.sort(key=lambda path: int(path.name.split("-")[1]))

    fd, assembled_name = tempfile.mkstemp(prefix=args.output.name + ".assembled-",
                                            dir=args.output.parent)
    assembled = Path(assembled_name)
    written = 0
    try:
        with os.fdopen(fd, "wb") as out:
            with args.prefix.open("rb") as source:
                shutil.copyfileobj(source, out, length=1024 * 1024)
            written += prefix_size
            for part in completed:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
                written += part.stat().st_size
            out.flush()
            os.fsync(out.fileno())
        if written != args.expected_size or assembled.stat().st_size != args.expected_size:
            raise RuntimeError(f"Assembled {written} bytes; expected {args.expected_size}")

        if args.output.exists():
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            backup = args.output.with_name(args.output.name + f".incomplete-prefix-{stamp}")
            if backup.exists():
                raise FileExistsError(backup)
            os.replace(args.output, backup)
            print(f"Preserved previous prefix as {backup}", flush=True)
        os.replace(assembled, args.output)
        print(f"Complete archive assembled: {args.output} ({args.expected_size} bytes)")
    finally:
        if assembled.exists():
            assembled.unlink()


if __name__ == "__main__":
    main()
