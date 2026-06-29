#!/usr/bin/env python3
"""Run deterministic C inference and validate recurrence against fixtures."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--gate-steps", type=int, default=3)
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def main() -> int:
    args = parse_args()
    exe = require_file(args.exe, "voxcpm-c executable")
    model = require_file(args.model, "model")
    fixtures = args.fixtures.resolve()
    if not fixtures.is_dir():
        raise FileNotFoundError(f"fixtures directory not found: {fixtures}")

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    for dump in work_dir.glob("dump_*.bin"):
        dump.unlink()
    for generated in ("c_latent_dump.bin", "recurrence.wav"):
        path = work_dir / generated
        if path.exists():
            path.unlink()

    env = os.environ.copy()
    env["VCPM_CFM_FIXTURE_DIR"] = str(fixtures)
    env["VCPM_DEBUG_SHAPES"] = "1"
    command = [
        str(exe),
        "tts",
        "--model",
        str(model),
        "--text",
        "Hello world.",
        "--out",
        str(work_dir / "recurrence.wav"),
        "--steps",
        "10",
        "--cfg",
        "2.0",
        "--seed",
        "1234",
        "--min-len",
        str(args.steps),
        "--max-len",
        str(args.steps),
        "--backend",
        "cpu",
    ]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    stdout_log = work_dir / "inference.stdout.log"
    stderr_log = work_dir / "inference.stderr.log"
    with stdout_log.open("w", encoding="utf-8") as stdout_stream, \
            stderr_log.open("w", encoding="utf-8") as stderr_stream:
        completed = subprocess.run(
            command,
            cwd=work_dir,
            env=env,
            stdout=stdout_stream,
            stderr=stderr_stream,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        print(stderr_log.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
        raise subprocess.CalledProcessError(completed.returncode, command)

    validator = Path(__file__).with_name("validate_recurrence_parity.py")
    validate_command = [
        sys.executable,
        str(validator),
        "--dump-dir",
        str(work_dir),
        "--fixtures",
        str(fixtures),
        "--steps",
        str(args.steps),
        "--gate-steps",
        str(args.gate_steps),
    ]
    return subprocess.run(validate_command, check=False).returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, OSError, subprocess.CalledProcessError) as exc:
        print(f"recurrence parity runner failed: {exc}", file=sys.stderr)
        sys.exit(2)
