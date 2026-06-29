#!/usr/bin/env python3
"""Export a deterministic upstream AudioVAE encoder parity fixture.

Python is used only to generate test data. The C runtime does not depend on it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def array_summary(value: np.ndarray) -> dict[str, object]:
    flat = value.astype(np.float64, copy=False).reshape(-1)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "mean": float(flat.mean()),
        "rms": float(np.sqrt(np.mean(np.square(flat)))),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path(r"D:\VoxCPM"),
        help="Upstream VoxCPM source checkout",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("pretrained_models/VoxCPM2"),
        help="Directory containing audiovae.pth",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("fixtures/ref/config.json"),
        help="VoxCPM2 config.json containing audio_vae_config",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("fixtures/ref"),
        help="Fixture output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    upstream_src = (args.upstream / "src").resolve()
    model_dir = args.model_dir.resolve()
    config_path = args.config.resolve()
    out_dir = args.out_dir.resolve()

    if not upstream_src.is_dir():
        raise FileNotFoundError(f"upstream source directory not found: {upstream_src}")
    checkpoint_path = model_dir / "audiovae.pth"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"AudioVAE checkpoint not found: {checkpoint_path}")

    sys.path.insert(0, str(upstream_src))
    from voxcpm.modules.audiovae.audio_vae_v2 import AudioVAE, AudioVAEConfig

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    vae_config = AudioVAEConfig.model_validate(raw_config["audio_vae_config"])

    torch.manual_seed(0)
    model = AudioVAE(config=vae_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    sample_rate = int(vae_config.sample_rate)
    time = torch.arange(sample_rate, dtype=torch.float32) / float(sample_rate)
    audio = (0.08 * torch.sin(2.0 * torch.pi * 220.0 * time)).reshape(1, 1, -1)
    with torch.inference_mode():
        mean = model.encode(audio, sample_rate=sample_rate)

    audio_np = np.ascontiguousarray(audio.numpy(), dtype=np.float32)
    mean_np = np.ascontiguousarray(mean.numpy(), dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vae_encoder_sine_input.npy", audio_np, allow_pickle=False)
    np.save(out_dir / "vae_encoder_sine_mu.npy", mean_np, allow_pickle=False)

    metadata = {
        "source": "upstream VoxCPM AudioVAE.encode",
        "checkpoint": checkpoint_path.name,
        "sample_rate": sample_rate,
        "frequency_hz": 220.0,
        "amplitude": 0.08,
        "audio": array_summary(audio_np),
        "mean": array_summary(mean_np),
    }
    (out_dir / "vae_encoder_sine_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
