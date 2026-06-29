#!/usr/bin/env python3
"""Export deterministic upstream AudioVAE clone-role fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def summary(value: np.ndarray) -> dict[str, object]:
    flat = value.astype(np.float64, copy=False).reshape(-1)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "mean": float(flat.mean()),
        "rms": float(np.sqrt(np.mean(np.square(flat)))),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path(r"D:\VoxCPM"))
    parser.add_argument(
        "--model-dir", type=Path, default=Path("pretrained_models/VoxCPM2")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("fixtures/ref/config.json")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("fixtures/ref"))
    return parser.parse_args()


def main() -> int:
    options = args()
    upstream_src = (options.upstream / "src").resolve()
    model_dir = options.model_dir.resolve()
    config_path = options.config.resolve()
    out_dir = options.out_dir.resolve()
    sys.path.insert(0, str(upstream_src))

    from voxcpm.modules.audiovae.audio_vae_v2 import AudioVAE, AudioVAEConfig

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = AudioVAEConfig.model_validate(raw_config["audio_vae_config"])
    checkpoint = torch.load(
        model_dir / "audiovae.pth", map_location="cpu", weights_only=True
    )
    model = AudioVAE(config=config)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    model.eval()

    n_samples = 16001
    time = torch.arange(n_samples, dtype=torch.float32) / float(config.sample_rate)
    audio = (0.08 * torch.sin(2.0 * torch.pi * 220.0 * time)).reshape(1, 1, -1)
    patch_len = int(model.chunk_size) * 4
    padding = (-n_samples) % patch_len
    right = torch.nn.functional.pad(audio, (0, padding))
    left = torch.nn.functional.pad(audio, (padding, 0))
    with torch.inference_mode():
        right_mu = model.encode(right, sample_rate=int(config.sample_rate))
        left_mu = model.encode(left, sample_rate=int(config.sample_rate))

    arrays = {
        "clone_sine_input.npy": audio.numpy(),
        "clone_sine_right_audio.npy": right.numpy(),
        "clone_sine_left_audio.npy": left.numpy(),
        "clone_sine_right_latent.npy": right_mu.numpy(),
        "clone_sine_left_latent.npy": left_mu.numpy(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "source": "upstream VoxCPM2 _encode_wav role semantics",
        "sample_rate": int(config.sample_rate),
        "patch_size": 4,
        "vae_hop_length": int(model.chunk_size),
        "patch_len": patch_len,
        "input_samples": n_samples,
        "padding_samples": padding,
        "arrays": {},
    }
    for name, value in arrays.items():
        contiguous = np.ascontiguousarray(value, dtype=np.float32)
        np.save(out_dir / name, contiguous, allow_pickle=False)
        metadata["arrays"][name] = summary(contiguous)
    (out_dir / "clone_fixture_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
