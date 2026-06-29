#!/usr/bin/env python3
"""Validate cross-patch VoxCPM recurrence dumps against Python fixtures."""

from __future__ import annotations

import argparse
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Boundary:
    name: str
    dump_name: str
    fixture_name: str
    transpose_patch: bool = False


def load_dump(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError(f"{path}: truncated dump header")
        ne0, ne1, ne2 = struct.unpack("<iii", header)
        data = np.frombuffer(stream.read(), dtype="<f4").astype(np.float32)
    expected = ne0 * max(ne1, 1) * max(ne2, 1)
    if data.size != expected:
        raise ValueError(
            f"{path}: header expects {expected} float32 values, found {data.size}"
        )
    return data


def load_fixture(path: Path, transpose_patch: bool) -> np.ndarray:
    data = np.load(path).astype(np.float32)
    if transpose_patch:
        if data.ndim != 3:
            raise ValueError(f"{path}: expected [batch, patch, latent] fixture")
        data = data.transpose(0, 2, 1)
    return data.reshape(-1)


def metrics(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    if actual.size != expected.size:
        raise ValueError(
            f"tensor size mismatch: C={actual.size}, Python={expected.size}"
        )
    actual64 = actual.astype(np.float64)
    expected64 = expected.astype(np.float64)
    denominator = np.linalg.norm(actual64) * np.linalg.norm(expected64)
    cosine = 1.0 if denominator == 0.0 else float(
        np.dot(actual64, expected64) / denominator
    )
    rmse = float(np.sqrt(np.mean((actual64 - expected64) ** 2)))
    return cosine, rmse


def boundaries(
    step: int, prompt_len: int, include_next_dit: bool
) -> list[Boundary]:
    fill_pos = prompt_len + step
    result = [
        Boundary(
            "pred_feat",
            f"dump_step_pred_feat_{step:04d}.bin",
            f"step{step:04d}_cfm_pred_feat.npy",
            True,
        ),
        Boundary(
            "locenc",
            f"dump_fe_output_update_{fill_pos:04d}.bin",
            f"step{step:04d}_curr_embed_raw.npy",
        ),
        Boundary(
            "audio_proj",
            f"dump_audio_embed_update_{fill_pos:04d}.bin",
            f"step{step:04d}_curr_embed_proj.npy",
        ),
        Boundary(
            "base_lm",
            f"dump_base_hidden_update_{fill_pos:04d}.bin",
            f"step{step:04d}_lm_hidden_step.npy",
        ),
        Boundary(
            "fsq",
            f"dump_lm_hidden_step_{fill_pos:04d}.bin",
            f"step{step:04d}_lm_hidden_fsq.npy",
        ),
        Boundary(
            "residual_lm",
            f"dump_residual_hidden_step_{fill_pos:04d}.bin",
            f"step{step:04d}_residual_hidden_step.npy",
        ),
    ]
    if include_next_dit:
        result.append(
            Boundary(
                "next_dit",
                f"dump_mu_init_{step + 1:04d}.bin",
                f"step{step + 1:04d}_dit_hidden.npy",
            )
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--prompt-len", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--gate-steps",
        type=int,
        default=3,
        help="Number of leading AR steps gated against CUDA fixtures",
    )
    parser.add_argument("--min-cosine", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.gate_steps < 1 or args.gate_steps > args.steps:
        raise ValueError("--gate-steps must be between 1 and --steps")
    if not -1.0 <= args.min_cosine <= 1.0:
        raise ValueError("--min-cosine must be between -1 and 1")

    failures: list[str] = []
    print("step boundary       cosine       rmse status")
    print("---- ------------- --------- ---------- ------")
    for step in range(args.steps):
        for boundary in boundaries(
            step, args.prompt_len, include_next_dit=step + 1 < args.steps
        ):
            dump_path = args.dump_dir / boundary.dump_name
            fixture_path = args.fixtures / boundary.fixture_name
            if not dump_path.is_file() or not fixture_path.is_file():
                missing = dump_path if not dump_path.is_file() else fixture_path
                failures.append(f"missing {missing}")
                print(f"{step:4d} {boundary.name:13s} {'-':>9s} {'-':>10s} MISS")
                continue
            try:
                actual = load_dump(dump_path)
                expected = load_fixture(fixture_path, boundary.transpose_patch)
                cosine, rmse = metrics(actual, expected)
            except (OSError, ValueError) as exc:
                failures.append(f"{boundary.name} step {step}: {exc}")
                print(f"{step:4d} {boundary.name:13s} {'-':>9s} {'-':>10s} ERROR")
                continue

            finite = math.isfinite(cosine) and math.isfinite(rmse)
            gated = step < args.gate_steps
            passed = finite and (not gated or cosine >= args.min_cosine)
            status = "OK" if passed and gated else "DIAG" if passed else "FAIL"
            print(
                f"{step:4d} {boundary.name:13s} "
                f"{cosine:9.6f} {rmse:10.6f} {status}"
            )
            if not passed:
                if not finite:
                    failures.append(
                        f"{boundary.name} step {step}: non-finite metric"
                    )
                else:
                    failures.append(
                        f"{boundary.name} step {step}: cosine {cosine:.6f} "
                        f"< {args.min_cosine:.6f}"
                    )

    if failures:
        print("\nRecurrence parity failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        "\nRecurrence parity passed "
        f"({args.gate_steps} CUDA-stable steps gated; later steps diagnostic)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
