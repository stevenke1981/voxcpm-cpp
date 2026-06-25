#!/usr/bin/env python3
"""
Create a minimal synthetic VoxCPM2 GGUF fixture for C-side model loader testing.

Creates a tiny GGUF with voxcpm2 architecture metadata and minimal tensors.
"""

import numpy as np
from gguf import GGUFWriter
from pathlib import Path
import argparse


def create_minimal_gguf(output_path: str) -> None:
    """Create a minimal but valid VoxCPM2 GGUF fixture."""
    writer = GGUFWriter(output_path, arch="voxcpm2")

    # ---- Required metadata ----
    writer.add_string("general.name", "VoxCPM2-Mini")
    writer.add_int32("voxcpm.version", 2)
    writer.add_int32("voxcpm.patch_size", 12)
    writer.add_int32("voxcpm.feat_dim", 64)
    writer.add_int32("voxcpm.latent_dim", 16)
    writer.add_int32("voxcpm.max_length", 512)

    # Sample rates
    writer.add_int32("voxcpm.sample_rate", 48000)
    writer.add_int32("voxcpm.encode_sample_rate", 16000)

    # Special tokens
    writer.add_int32("voxcpm.audio_start_token", 101)
    writer.add_int32("voxcpm.audio_end_token", 102)
    writer.add_int32("voxcpm.ref_audio_start_token", 103)
    writer.add_int32("voxcpm.ref_audio_end_token", 104)

    # Capability flags
    writer.add_bool("voxcpm.supports_reference_audio", True)
    writer.add_bool("voxcpm.supports_streaming", True)

    # ---- MiniCPM4 (base_lm) config ----
    writer.add_int32("voxcpm.hidden_size", 256)
    writer.add_int32("voxcpm.num_hidden_layers", 4)
    writer.add_int32("voxcpm.num_attention_heads", 8)
    writer.add_int32("voxcpm.num_kv_heads", 4)
    writer.add_int32("voxcpm.intermediate_size", 1024)
    writer.add_int32("voxcpm.head_dim", 32)
    writer.add_float32("voxcpm.rms_norm_eps", 1.0e-6)
    writer.add_int32("voxcpm.rope_theta", 1000000)
    writer.add_int32("voxcpm.max_seq_len", 512)

    # ---- Residual LM config ----
    writer.add_int32("voxcpm.res_hidden_size", 256)
    writer.add_int32("voxcpm.res_num_layers", 2)
    writer.add_int32("voxcpm.res_num_heads", 8)
    writer.add_int32("voxcpm.res_num_kv_heads", 4)

    # ---- AudioVAE config ----
    writer.add_int32("voxcpm.vae_latent_dim", 16)
    writer.add_int32("voxcpm.vae_sample_rate", 16000)
    writer.add_int32("voxcpm.vae_out_sample_rate", 48000)

    # ---- LocDiT config ----
    writer.add_int32("voxcpm.dit_hidden_size", 128)
    writer.add_int32("voxcpm.dit_num_layers", 4)
    writer.add_int32("voxcpm.dit_num_heads", 4)

    # ---- Tokenizer metadata ----
    token_strings = [
        "<unk>", "<s>", "</s>",
        "the", "a", "an",
        "hello", "world", "test",
        "你", "好", "世", "界",
    ]
    writer.add_token_list(token_strings)
    writer.add_token_scores([0.0] * len(token_strings))
    writer.add_token_types([0] * len(token_strings))
    writer.add_tokenizer_model("llama")
    writer.add_int32("tokenizer.ggml.bos_token_id", 1)
    writer.add_int32("tokenizer.ggml.eos_token_id", 2)
    writer.add_int32("voxcpm.vocab_size", len(token_strings))

    # ---- Tensors ----
    # Random weights for a tiny model
    rs = np.random.RandomState(42)  # deterministic random

    def rand(*shape):
        return rs.randn(*shape).astype(np.float32)

    def zeros(*shape):
        return np.zeros(shape, dtype=np.float32)

    # Base LM
    writer.add_tensor("base_lm.embed_tokens.weight", rand(256, 13))
    writer.add_tensor("base_lm.layers.0.self_attn.q_proj.weight", rand(256, 256))
    writer.add_tensor("base_lm.layers.0.self_attn.k_proj.weight", rand(128, 256))
    writer.add_tensor("base_lm.layers.0.self_attn.v_proj.weight", rand(128, 256))
    writer.add_tensor("base_lm.layers.0.self_attn.o_proj.weight", rand(256, 256))
    writer.add_tensor("base_lm.layers.0.mlp.gate_proj.weight", rand(1024, 256))
    writer.add_tensor("base_lm.layers.0.mlp.up_proj.weight", rand(1024, 256))
    writer.add_tensor("base_lm.layers.0.mlp.down_proj.weight", rand(256, 1024))
    writer.add_tensor("base_lm.layers.0.input_layernorm.weight", rand(256))
    writer.add_tensor("base_lm.layers.0.post_attention_layernorm.weight", rand(256))
    writer.add_tensor("base_lm.norm.weight", rand(256))

    # Residual LM
    writer.add_tensor("residual_lm.layers.0.self_attn.q_proj.weight", rand(256, 256))
    writer.add_tensor("residual_lm.layers.0.self_attn.o_proj.weight", rand(256, 256))
    writer.add_tensor("residual_lm.layers.0.mlp.gate_proj.weight", rand(1024, 256))
    writer.add_tensor("residual_lm.layers.0.mlp.down_proj.weight", rand(256, 1024))

    # Projections
    writer.add_tensor("enc_to_lm_proj.weight", rand(256, 64))
    writer.add_tensor("lm_to_dit_proj.weight", rand(128, 256))
    writer.add_tensor("res_to_dit_proj.weight", rand(128, 256))

    # Stop predictor
    writer.add_tensor("stop_proj.weight", rand(1, 256))
    writer.add_tensor("stop_proj.bias", zeros(1))

    # Write the file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    import os
    size = os.path.getsize(output_path)
    print(f"Created: {output_path}")
    print(f"  Tensors: {len(writer.tensors)}")
    print(f"  File size: {size} bytes ({size/1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a minimal synthetic VoxCPM2 GGUF fixture.")
    parser.add_argument(
        "output",
        nargs="?",
        default=Path(__file__).with_name("minimal.gguf"),
        help="Output GGUF path (default: tests/fixtures/minimal.gguf)",
    )
    args = parser.parse_args()
    create_minimal_gguf(str(args.output))
