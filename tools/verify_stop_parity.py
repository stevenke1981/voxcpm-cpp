#!/usr/bin/env python3
"""Verify the GGUF stop head against exported Python fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

import gguf
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--fixtures", type=Path, default=Path("fixtures/ref")
    )
    parser.add_argument("--max-abs", type=float, default=0.04)
    args = parser.parse_args()

    reader = gguf.GGUFReader(str(args.model))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    required = ("stop_proj.weight", "stop_proj.bias", "stop_head.weight")
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"missing GGUF tensors: {', '.join(missing)}")

    proj_weight = np.asarray(
        tensors["stop_proj.weight"].data, dtype=np.float32
    )
    proj_bias = np.asarray(
        tensors["stop_proj.bias"].data, dtype=np.float32
    ).reshape(-1)
    head_weight = np.asarray(
        tensors["stop_head.weight"].data, dtype=np.float32
    )

    failures = 0
    for step in range(1, 5):
        # stop at step N consumes the FSQ hidden produced after step N-1.
        hidden = np.load(
            args.fixtures / f"step{step - 1:04d}_lm_hidden_fsq.npy"
        ).reshape(-1).astype(np.float32)
        expected = np.load(
            args.fixtures / f"step{step:04d}_stop_logits.npy"
        ).reshape(-1).astype(np.float32)

        projected = proj_weight @ hidden + proj_bias
        stop_hidden = projected / (1.0 + np.exp(-projected))
        actual = head_weight @ stop_hidden
        max_abs = float(np.max(np.abs(actual - expected)))
        actual_class = int(np.argmax(actual))
        expected_class = int(np.argmax(expected))
        ok = max_abs <= args.max_abs and actual_class == expected_class
        failures += int(not ok)
        print(
            f"step={step} actual={actual.tolist()} "
            f"expected={expected.tolist()} max_abs={max_abs:.6f} "
            f"class={actual_class} {'PASS' if ok else 'FAIL'}"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
