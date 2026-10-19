#!/usr/bin/env python3
"""List and resumably download a publicly shared Google Drive folder."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import threading
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

FOLDER_MIME = "application/vnd.google-apps.folder"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"
LOCAL = threading.local()


def get_session() -> requests.Session:
    if not hasattr(LOCAL, "session"):
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        LOCAL.session = session
    return LOCAL.session


def folder_id_from_url(url: str) -> str:
    match = re.search(r"/folders/([A-Za-z0-9_-]+)", url)
    if not match:
        raise ValueError("Expected a Google Drive shared folder URL")
    return match.group(1)


def public_api_key(url: str, folder_id: str, timeout: int) -> str:
    page = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    page.raise_for_status()
    candidates = list(dict.fromkeys(re.findall(r"AIza[0-9A-Za-z_-]{30,}", page.text)))
    params = {"q": f"'{folder_id}' in parents and trashed=false",
              "fields": "files(id)", "pageSize": 1}
    for key in candidates:
        params["key"] = key
        response = requests.get("https://www.googleapis.com/drive/v3/files",
                                params=params, timeout=timeout)
        if response.ok and isinstance(response.json().get("files"), list):
            return key
    raise RuntimeError("Could not find an anonymous Drive listing key on the public folder page")


def list_children(folder_id: str, key: str, timeout: int) -> list[dict]:
    params = {"q": f"'{folder_id}' in parents and trashed=false",
              "fields": "nextPageToken,files(id,name,mimeType,size,md5Checksum)",
              "pageSize": 1000, "key": key}
    result = []
    while True:
        response = requests.get("https://www.googleapis.com/drive/v3/files",
                                params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        result.extend(payload.get("files", []))
        token = payload.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
    return result


def safe_component(name: str) -> str:
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\0" in name:
        raise ValueError(f"Unsafe Google Drive item name: {name!r}")
    return name


def discover(folder_id: str, key: str, timeout: int) -> list[dict]:
    found: list[dict] = []
    pending = [(folder_id, PurePosixPath("."))]
    visited: set[str] = set()
    while pending:
        parent_id, parent_path = pending.pop()
        if parent_id in visited:
            raise ValueError(f"Repeated/cyclic folder ID: {parent_id}")
        visited.add(parent_id)
        for item in list_children(parent_id, key, timeout):
            name = safe_component(item["name"])
            relative = parent_path / name
            if item.get("mimeType") == FOLDER_MIME:
                pending.append((item["id"], relative))
            else:
                item["relative_path"] = relative.as_posix()
                found.append(item)
    paths = [item["relative_path"] for item in found]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate paths in public Drive folder")
    return sorted(found, key=lambda item: item["relative_path"].casefold())


def fetch_file(item: dict, output: Path, timeout: int, attempts: int) -> dict:
    relative = PurePosixPath(item["relative_path"])
    target = output.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    expected_size = int(item.get("size", -1))
    expected_md5 = item.get("md5Checksum")
    if target.is_file() and (expected_size < 0 or target.stat().st_size == expected_size):
        if expected_md5:
            digest = hashlib.md5()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest().lower() == expected_md5.lower():
                return {"relative_path": item["relative_path"], "bytes": target.stat().st_size,
                        "md5_verified": True, "already_present": True}
        elif expected_size >= 0:
            return {"relative_path": item["relative_path"], "bytes": target.stat().st_size,
                    "md5_verified": False, "already_present": True}
    for attempt in range(1, attempts + 1):
        current = partial.stat().st_size if partial.exists() else 0
        if expected_size >= 0 and current > expected_size:
            partial.unlink()
            current = 0
        headers = {"Range": f"bytes={current}-"} if current else {}
        try:
            url = "https://drive.usercontent.google.com/download"
            response = get_session().get(url, params={"id": item["id"], "export": "download"},
                                         headers=headers, stream=True, timeout=(30, timeout))
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                raise RuntimeError("Drive returned an HTML confirmation/error page instead of file bytes")
            mode = "ab" if current and response.status_code == 206 else "wb"
            if mode == "wb":
                current = 0
            with partial.open(mode) as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        stream.write(chunk)
            response.close()
            size = partial.stat().st_size
            if expected_size >= 0 and size != expected_size:
                raise RuntimeError(f"size {size} differs from Drive metadata {expected_size}")
            if expected_md5:
                digest = hashlib.md5()
                with partial.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest().lower() != expected_md5.lower():
                    raise RuntimeError("MD5 differs from Drive metadata")
            partial.replace(target)
            return {"relative_path": item["relative_path"], "bytes": size,
                    "md5_verified": bool(expected_md5)}
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Failed to download {item['relative_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder_url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if not urlparse(args.folder_url).hostname or args.workers < 1 or args.attempts < 1:
        parser.error("provide a valid folder URL, positive worker count, and positive attempts")

    root_id = folder_id_from_url(args.folder_url)
    key = public_api_key(args.folder_url, root_id, args.timeout)
    items = discover(root_id, key, args.timeout)
    report = {"folder_url": args.folder_url, "root_folder_id": root_id,
              "file_count": len(items),
              "total_bytes": sum(int(item.get("size", 0)) for item in items),
              "files": items}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(f"Discovered {len(items)} files ({report['total_bytes']} bytes)")
    print(f"Manifest: {args.manifest}")
    if args.manifest_only:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_file, item, args.output, args.timeout, args.attempts)
                   for item in items]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(items):
                print(f"Downloaded and verified {completed}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
