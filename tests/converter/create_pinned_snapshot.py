#!/usr/bin/env python3
"""Create a tiny, deterministic snapshot derived from the pinned VoxCPM2 ABI.

The fixture uses the real upstream module/tensor names but deliberately tiny
dimensions, so CI exercises converter I/O without downloading model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


UPSTREAM_REPO = "openbmb/VoxCPM2"
UPSTREAM_REVISION = "bffb3df5a29440629464e5e839f4d214c8714c3d"


def deterministic(shape: tuple[int, ...], seed: int) -> np.ndarray:
    count = int(np.prod(shape))
    values = (np.arange(count, dtype=np.float32) + seed) / max(count, 1)
    return values.reshape(shape)


def add_transformer(
    tensors: dict[str, np.ndarray],
    prefix: str,
    hidden: int,
    intermediate: int,
    kv_size: int,
    seed: int,
) -> int:
    shapes = {
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
        "self_attn.q_proj.weight": (hidden, hidden),
        "self_attn.k_proj.weight": (kv_size, hidden),
        "self_attn.v_proj.weight": (kv_size, hidden),
        "self_attn.o_proj.weight": (hidden, hidden),
        "mlp.gate_proj.weight": (intermediate, hidden),
        "mlp.up_proj.weight": (intermediate, hidden),
        "mlp.down_proj.weight": (hidden, intermediate),
    }
    for suffix, shape in shapes.items():
        tensors[f"{prefix}.{suffix}"] = deterministic(shape, seed)
        seed += 1
    return seed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    hidden = 8
    intermediate = 16
    kv_size = 4
    vocab = 16
    config = {
        "architecture": "voxcpm2",
        "lm_config": {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "max_position_embeddings": 128,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000,
            "kv_channels": 4,
            "vocab_size": vocab,
            "use_mup": False,
            "scale_depth": 1.4,
        },
        "patch_size": 4,
        "feat_dim": 4,
        "scalar_quantization_latent_dim": 8,
        "scalar_quantization_scale": 9,
        "residual_lm_num_layers": 1,
        "residual_lm_no_rope": True,
        "encoder_config": {
            "hidden_dim": hidden,
            "ffn_dim": intermediate,
            "num_heads": 2,
            "num_layers": 1,
            "kv_channels": 4,
        },
        "dit_config": {
            "hidden_dim": hidden,
            "ffn_dim": intermediate,
            "num_heads": 2,
            "num_layers": 1,
            "kv_channels": 4,
            "cfm_config": {
                "sigma_min": 1e-6,
                "solver": "euler",
                "inference_cfg_rate": 2.0,
            },
        },
        "audio_vae_config": {
            "encoder_dim": 4,
            "latent_dim": 4,
            "decoder_dim": 8,
            "sample_rate": 16000,
            "out_sample_rate": 48000,
        },
        "max_length": 128,
    }
    config_path = out / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    tokenizer = {
        "model": {
            "type": "BPE",
            "vocab": {f"<token_{i}>": i for i in range(vocab)},
            "merges": [],
        },
        "added_tokens": [
            {"id": 1, "content": "<s>", "special": True},
            {"id": 2, "content": "</s>", "special": True},
        ],
    }
    tokenizer_path = out / "tokenizer.json"
    tokenizer_path.write_text(
        json.dumps(tokenizer, indent=2) + "\n", encoding="utf-8"
    )

    tensors: dict[str, np.ndarray] = {
        "base_lm.embed_tokens.weight": deterministic((vocab, hidden), 1),
        "base_lm.norm.weight": np.ones((hidden,), dtype=np.float32),
        "residual_lm.norm.weight": np.ones((hidden,), dtype=np.float32),
        "feat_encoder.in_proj.weight": deterministic((hidden, 4), 2),
        "feat_encoder.in_proj.bias": np.zeros((hidden,), dtype=np.float32),
        "feat_encoder.special_token": deterministic((hidden,), 3),
        "feat_encoder.encoder.norm.weight": np.ones(
            (hidden,), dtype=np.float32
        ),
        "feat_decoder.estimator.in_proj.weight": deterministic(
            (hidden, hidden), 4
        ),
        "feat_decoder.estimator.out_proj.weight": deterministic(
            (4, hidden), 5
        ),
        "feat_decoder.estimator.cond_proj.weight": deterministic(
            (hidden, hidden), 6
        ),
        "feat_decoder.estimator.decoder.norm.weight": np.ones(
            (hidden,), dtype=np.float32
        ),
        "fsq_layer.in_proj.weight": deterministic((8, 4), 7),
        "fsq_layer.out_proj.weight": deterministic((4, 8), 8),
        "fsq_layer.scale": np.ones((8,), dtype=np.float32),
        "fsq_layer.offset": np.zeros((8,), dtype=np.float32),
        "enc_to_lm_proj.weight": deterministic((hidden, hidden), 9),
        "lm_to_dit_proj.weight": deterministic((hidden, hidden), 10),
        "res_to_dit_proj.weight": deterministic((hidden, hidden), 11),
        "fusion_concat_proj.weight": deterministic((hidden, hidden * 2), 12),
        "stop_proj.weight": deterministic((hidden, hidden), 13),
        "stop_head.weight": deterministic((1, hidden), 14),
    }
    seed = 20
    seed = add_transformer(
        tensors, "base_lm.layers.0", hidden, intermediate, kv_size, seed
    )
    seed = add_transformer(
        tensors, "residual_lm.layers.0", hidden, intermediate, kv_size, seed
    )
    seed = add_transformer(
        tensors,
        "feat_encoder.encoder.layers.0",
        hidden,
        intermediate,
        kv_size,
        seed,
    )
    add_transformer(
        tensors,
        "feat_decoder.estimator.decoder.layers.0",
        hidden,
        intermediate,
        kv_size,
        seed,
    )
    model_path = out / "model.safetensors"
    save_file(tensors, model_path)

    lock = {
        "schema_version": 1,
        "snapshot_kind": "synthetic-contract",
        "derived_from": {
            "repo_id": UPSTREAM_REPO,
            "revision": UPSTREAM_REVISION,
        },
        "files": {
            "config.json": sha256(config_path),
            "tokenizer.json": sha256(tokenizer_path),
            "model.safetensors": sha256(model_path),
        },
    }
    (out / "snapshot.lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    print(f"created deterministic converter snapshot: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
