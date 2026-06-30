#!/usr/bin/env python3
"""Weight-free end-to-end converter and GGUF contract regression."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(*args: str, expect: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != expect:
        raise AssertionError(
            f"command returned {result.returncode}, expected {expect}:\n"
            f"{' '.join(args)}\n{result.stdout}"
        )
    return result


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="voxcpm-converter-") as temp:
        work = Path(temp)
        snapshot = work / "snapshot"
        output = work / "model.gguf"
        manifest = work / "shapes.json"

        run(
            "tests/converter/create_pinned_snapshot.py",
            "--out-dir",
            str(snapshot),
        )
        run(
            "tools/convert_voxcpm2_to_gguf.py",
            "--hf-dir",
            str(snapshot),
            "--out",
            str(output),
            "--outtype",
            "f16",
            "--snapshot-lock",
            str(snapshot / "snapshot.lock.json"),
            "--emit-shape-manifest",
            str(manifest),
        )
        run(
            "tools/validate_gguf_contract.py",
            "--gguf",
            str(output),
            "--shape-manifest",
            str(manifest),
        )

        shape_data = json.loads(manifest.read_text(encoding="utf-8"))
        assert shape_data["base_lm.blk.0.self_attn.k_proj.weight"][
            "gguf_shape"
        ] == [8, 4]
        assert "residual_lm.blk.0.mlp.down_proj.weight" in shape_data
        assert "feat_encoder.blk.0.self_attn.q_proj.weight" in shape_data
        assert "feat_decoder.estimator.blk.0.self_attn.q_proj.weight" in shape_data

        lock_path = snapshot / "snapshot.lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["files"]["config.json"] = "0" * 64
        bad_lock = work / "bad.lock.json"
        bad_lock.write_text(
            json.dumps(lock, indent=2) + "\n", encoding="utf-8"
        )
        rejected = run(
            "tools/convert_voxcpm2_to_gguf.py",
            "--hf-dir",
            str(snapshot),
            "--out",
            str(work / "bad.gguf"),
            "--snapshot-lock",
            str(bad_lock),
            expect=1,
        )
        assert "SHA-256 mismatch" in rejected.stdout

    print("PASS: pinned converter and GGUF contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
