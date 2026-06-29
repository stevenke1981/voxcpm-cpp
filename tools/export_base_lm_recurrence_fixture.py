#!/usr/bin/env python3
"""Export CPU/BF16 Base LM recurrence fixtures from upstream VoxCPM2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-src", type=Path, required=True)
    parser.add_argument("--safetensors", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.upstream_src.resolve()))
    from voxcpm.model.voxcpm2 import VoxCPMConfig
    from voxcpm.modules.minicpm4.model import MiniCPMModel

    config = VoxCPMConfig.model_validate_json(
        args.config.read_text(encoding="utf-8")
    )
    model = MiniCPMModel(config.lm_config)
    state = {}
    with safe_open(
        str(args.safetensors.resolve()), framework="pt", device="cpu"
    ) as source:
        for key in source.keys():
            if key.startswith("base_lm."):
                state[key.removeprefix("base_lm.")] = source.get_tensor(key)
    model.load_state_dict(state, strict=True)
    del state

    model = model.to(torch.bfloat16).eval()
    model.setup_cache(
        batch_size=1,
        max_length=8192,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    fixtures = args.fixtures.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_fixture in out_dir.glob("step*_lm_hidden_step.npy"):
        old_fixture.unlink()

    cosines_to_cuda: list[float] = []
    with torch.inference_mode():
        prompt = torch.from_numpy(
            np.load(fixtures / "combined_embed.npy")
        ).to(torch.bfloat16)
        _, prompt_cache = model(prompt, is_causal=True)
        model.kv_cache.fill_caches(prompt_cache)

        for step in range(args.steps):
            current = torch.from_numpy(
                np.load(fixtures / f"step{step:04d}_curr_embed_proj.npy")[
                    :, 0, :
                ]
            ).to(torch.bfloat16)
            position = torch.tensor([model.kv_cache.step()])
            hidden = model.forward_step(current, position).clone()
            output = hidden.float().numpy()
            np.save(out_dir / f"step{step:04d}_lm_hidden_step.npy", output)

            cuda_reference = torch.from_numpy(
                np.load(fixtures / f"step{step:04d}_lm_hidden_step.npy")
            ).float()
            cosine = torch.nn.functional.cosine_similarity(
                hidden.float(), cuda_reference
            ).item()
            cosines_to_cuda.append(cosine)
            print(f"step {step}: CPU vs CUDA fixture cosine={cosine:.9f}")

    manifest = {
        "source": "OpenBMB/VoxCPM MiniCPMModel.forward_step",
        "device": "cpu",
        "dtype": "bfloat16",
        "torch_version": torch.__version__,
        "steps": args.steps,
        "teacher_forcing_inputs": "fixtures/ref curr_embed_proj",
        "cuda_reference_cosines": cosines_to_cuda,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
