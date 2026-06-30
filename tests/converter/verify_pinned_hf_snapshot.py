#!/usr/bin/env python3
"""Verify small files from the exact Hugging Face revision used by CI."""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "tests/converter/hf_snapshot.lock.json"


def main() -> int:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    source = lock["source"]
    repo = source["repo_id"]
    revision = source["revision"]
    if len(revision) != 40:
        raise ValueError("Hugging Face revision must be a full commit")

    with tempfile.TemporaryDirectory(prefix="voxcpm-hf-pin-") as temp:
        temp_dir = Path(temp)
        for relative in lock["ci_files"]:
            url = (
                f"https://huggingface.co/{repo}/resolve/{revision}/"
                f"{relative}?download=true"
            )
            destination = temp_dir / relative
            request = urllib.request.Request(
                url, headers={"User-Agent": "voxcpm-c-converter-contract/1"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                destination.write_bytes(response.read())
            actual = hashlib.sha256(destination.read_bytes()).hexdigest()
            expected = lock["files"][relative]
            if actual != expected:
                raise ValueError(
                    f"SHA-256 mismatch for {relative}: "
                    f"expected {expected}, got {actual}"
                )

    print(f"PASS: pinned Hugging Face snapshot {repo}@{revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
