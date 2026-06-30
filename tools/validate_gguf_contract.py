#!/usr/bin/env python3
"""Validate VoxCPM2 GGUF metadata, tensor names, types, and shapes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import gguf


REQUIRED_METADATA = (
    "general.architecture",
    "voxcpm.source_repo",
    "voxcpm.source_revision",
    "voxcpm.patch_size",
    "voxcpm.feat_dim",
    "voxcpm.latent_dim",
    "voxcpm.hidden_size",
    "voxcpm.num_hidden_layers",
    "voxcpm.num_attention_heads",
    "voxcpm.num_kv_heads",
    "voxcpm.intermediate_size",
    "voxcpm.head_dim",
    "voxcpm.rms_norm_eps",
    "voxcpm.res_hidden_size",
    "voxcpm.res_num_layers",
    "voxcpm.res_num_heads",
    "voxcpm.res_num_kv_heads",
    "voxcpm.enc_hidden_size",
    "voxcpm.enc_num_layers",
    "voxcpm.enc_num_heads",
    "voxcpm.dit_hidden_size",
    "voxcpm.dit_num_layers",
    "voxcpm.dit_num_heads",
    "tokenizer.ggml.model",
    "tokenizer.ggml.tokens",
)

LAYER_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)


def fail(message: str) -> None:
    raise ValueError(message)


def metadata(reader: gguf.GGUFReader, key: str) -> Any:
    field = reader.fields.get(key)
    if field is None:
        fail(f"missing required metadata: {key}")
    return field.contents()


def require_positive_int(reader: gguf.GGUFReader, key: str) -> int:
    value = metadata(reader, key)
    if not isinstance(value, (int, float)) or int(value) <= 0:
        fail(f"metadata {key} must be a positive integer, got {value!r}")
    return int(value)


def validate_layer_names(
    tensors: dict[str, Any], prefix: str, layers: int
) -> None:
    for layer in range(layers):
        for suffix in LAYER_SUFFIXES:
            name = f"{prefix}.{layer}.{suffix}"
            if name not in tensors:
                fail(f"missing required tensor: {name}")


def validate_manifest(
    tensors: dict[str, Any], manifest_path: Path
) -> None:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(expected, dict) or not expected:
        fail("shape manifest must be a non-empty object")
    for name, contract in expected.items():
        tensor = tensors.get(name)
        if tensor is None:
            fail(f"manifest tensor is absent from GGUF: {name}")
        expected_shape = [int(x) for x in contract.get("gguf_shape", [])]
        actual_shape = [int(x) for x in tensor.shape]
        if actual_shape != expected_shape:
            fail(
                f"shape mismatch for {name}: "
                f"expected {expected_shape}, got {actual_shape}"
            )
        expected_type = contract.get("gguf_type")
        actual_type = tensor.tensor_type.name
        if expected_type and actual_type != expected_type:
            fail(
                f"type mismatch for {name}: "
                f"expected {expected_type}, got {actual_type}"
            )


def validate(path: Path, manifest_path: Path | None) -> None:
    reader = gguf.GGUFReader(str(path), "r")
    for key in REQUIRED_METADATA:
        metadata(reader, key)
    if metadata(reader, "general.architecture") != "voxcpm2":
        fail("general.architecture must equal 'voxcpm2'")
    if not metadata(reader, "voxcpm.source_repo"):
        fail("voxcpm.source_repo must not be empty")
    revision = str(metadata(reader, "voxcpm.source_revision"))
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        fail("voxcpm.source_revision must be a full lowercase Git commit")

    for key in (
        "voxcpm.patch_size",
        "voxcpm.feat_dim",
        "voxcpm.latent_dim",
        "voxcpm.hidden_size",
        "voxcpm.num_hidden_layers",
        "voxcpm.num_attention_heads",
        "voxcpm.num_kv_heads",
        "voxcpm.intermediate_size",
        "voxcpm.res_num_layers",
        "voxcpm.enc_num_layers",
        "voxcpm.dit_num_layers",
    ):
        require_positive_int(reader, key)
    eps = float(metadata(reader, "voxcpm.rms_norm_eps"))
    if not math.isfinite(eps) or eps <= 0.0:
        fail(f"voxcpm.rms_norm_eps must be finite and positive, got {eps}")

    tensor_list = list(reader.tensors)
    tensors = {tensor.name: tensor for tensor in tensor_list}
    if len(tensors) != len(tensor_list):
        fail("GGUF contains duplicate tensor names")
    for tensor in tensor_list:
        if (
            not tensor.name
            or len(tensor.shape) == 0
            or any(int(x) <= 0 for x in tensor.shape)
        ):
            fail(f"invalid tensor name or shape: {tensor.name!r} {tensor.shape}")

    validate_layer_names(
        tensors,
        "base_lm.blk",
        require_positive_int(reader, "voxcpm.num_hidden_layers"),
    )
    validate_layer_names(
        tensors,
        "residual_lm.blk",
        require_positive_int(reader, "voxcpm.res_num_layers"),
    )
    validate_layer_names(
        tensors,
        "feat_encoder.blk",
        require_positive_int(reader, "voxcpm.enc_num_layers"),
    )
    validate_layer_names(
        tensors,
        "feat_decoder.estimator.blk",
        require_positive_int(reader, "voxcpm.dit_num_layers"),
    )

    for name in (
        "base_lm.embed_tokens.weight",
        "base_lm.norm.weight",
        "residual_lm.norm.weight",
        "feat_encoder.in_proj.weight",
        "feat_encoder.special_token",
        "feat_encoder.norm.weight",
        "feat_decoder.estimator.in_proj.weight",
        "feat_decoder.estimator.out_proj.weight",
        "feat_decoder.estimator.cond_proj.weight",
        "feat_decoder.estimator.norm.weight",
        "enc_to_lm_proj.weight",
        "lm_to_dit_proj.weight",
        "res_to_dit_proj.weight",
        "fusion_concat_proj.weight",
    ):
        if name not in tensors:
            fail(f"missing required tensor: {name}")

    hidden = require_positive_int(reader, "voxcpm.hidden_size")
    vocab = require_positive_int(reader, "voxcpm.vocab_size")
    expected_embed = [hidden, vocab]
    actual_embed = [int(x) for x in tensors["base_lm.embed_tokens.weight"].shape]
    if actual_embed != expected_embed:
        fail(
            "shape mismatch for base_lm.embed_tokens.weight: "
            f"expected {expected_embed}, got {actual_embed}"
        )
    if [int(x) for x in tensors["base_lm.norm.weight"].shape] != [hidden]:
        fail("base_lm.norm.weight must match voxcpm.hidden_size")

    if manifest_path is not None:
        validate_manifest(tensors, manifest_path)

    print(
        f"PASS: GGUF contract ({len(tensors)} tensors, "
        f"{len(reader.fields)} metadata fields): {path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--shape-manifest", type=Path)
    args = parser.parse_args()
    try:
        validate(args.gguf, args.shape_manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
