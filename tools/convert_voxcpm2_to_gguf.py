#!/usr/bin/env python3
"""
VoxCPM2 Hugging Face snapshot → GGUF converter.

Reads a VoxCPM2 HF snapshot (config.json, tokenizer.json, model.safetensors,
audiovae.pth), maps tensor names to canonical GGUF names, and writes a
single .gguf file.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gguf
import numpy as np
import torch
from safetensors import safe_open

log = logging.getLogger(__name__)

# ---- Upstream → Canonical prefix mapping ----
# Each entry: (upstream_prefix_re, canonical_template)
# The first group is either a layer index (\\d+) or a suffix capture (weight|bias).
PREFIX_MAP: List[Tuple[str, str]] = [
    # === Base LM ===
    (r"^base_lm\.layers\.(\d+)\.(.*)$",                 "base_lm.blk.{}.{}"),
    (r"^base_lm\.(embed_tokens|norm)\.(weight)$",       "base_lm.{}.{}"),

    # === Residual LM (RALM) ===
    (r"^residual_lm\.layers\.(\d+)\.(.*)$",              "residual_lm.blk.{}.{}"),
    (r"^residual_lm\.(norm)\.(weight)$",                 "residual_lm.{}.{}"),

    # === Feat Encoder (LocEnc) ===
    (r"^feat_encoder\.encoder\.layers\.(\d+)\.(.*)$",    "feat_encoder.blk.{}.{}"),
    (r"^feat_encoder\.encoder\.(norm)\.(weight)$",       "feat_encoder.{}.{}"),
    (r"^feat_encoder\.(in_proj)\.(weight|bias)$",        "feat_encoder.{}.{}"),
    (r"^feat_encoder\.(special_token)$",                 "feat_encoder.{}"),

    # === Feat Decoder / Estimator (LocDiT) ===
    (r"^feat_decoder\.estimator\.decoder\.layers\.(\d+)\.(.*)$",
                                                         "feat_decoder.estimator.blk.{}.{}"),
    (r"^feat_decoder\.estimator\.decoder\.(norm)\.(weight)$",
                                                         "feat_decoder.estimator.{}.{}"),
    (r"^feat_decoder\.estimator\.(in_proj)\.(weight|bias)$",
                                                         "feat_decoder.estimator.{}.{}"),
    (r"^feat_decoder\.estimator\.(out_proj)\.(weight|bias)$",
                                                         "feat_decoder.estimator.{}.{}"),
    (r"^feat_decoder\.estimator\.(cond_proj)\.(weight|bias)$",
                                                         "feat_decoder.estimator.{}.{}"),
    (r"^feat_decoder\.estimator\.(time_mlp)\.(linear_1|linear_2)\.(weight|bias)$",
                                                         "feat_decoder.estimator.{}.{}.{}"),
    (r"^feat_decoder\.estimator\.(delta_time_mlp)\.(linear_1|linear_2)\.(weight|bias)$",
                                                         "feat_decoder.estimator.{}.{}.{}"),

    # === FSQ ===
    (r"^fsq_layer\.(in_proj|out_proj)\.(weight|bias)$", "fsq.{}.{}"),
    (r"^fsq_layer\.(scale|offset)$",                     "fsq.{}"),

    # === Projections ===
    (r"^(enc_to_lm_proj)\.(weight|bias)$",               "{}.{}"),
    (r"^(lm_to_dit_proj)\.(weight|bias)$",               "{}.{}"),
    (r"^(res_to_dit_proj)\.(weight|bias)$",              "{}.{}"),
    (r"^(fusion_concat_proj)\.(weight|bias)$",           "{}.{}"),

    # === Stop predictor ===
    (r"^(stop_proj)\.(weight|bias)$",                    "{}.{}"),
    (r"^(stop_head)\.(weight)$",                         "{}.{}"),

    # === Generic fallthrough (for unmapped but known prefixes) ===
    (r"^(base_lm)\.(.*)$",                               "{}.{}"),
    (r"^(residual_lm)\.(.*)$",                           "{}.{}"),
    (r"^(feat_encoder)\.(.*)$",                          "{}.{}"),
    (r"^(feat_decoder)\.(.*)$",                          "{}.{}"),
    (r"^(fsq_layer)\.(.*)$",                             "fsq.{}"),
]


def load_config(hf_dir: Path) -> dict:
    """Load config.json from HF snapshot and flatten nested keys."""
    cfg_path = hf_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    log.info("Loaded config.json: architecture=%s", cfg.get("architecture", "?"))

    # Flatten nested config groups to top-level keys for simpler metadata extraction
    flat = dict(cfg)
    # lm_config → voxcpm.*
    lm_cfg = cfg.get("lm_config", {})
    for k, v in lm_cfg.items():
        flat[f"lm_{k}"] = v
    # encoder_config → voxcpm.encoder.*
    enc_cfg = cfg.get("encoder_config", {})
    if enc_cfg:
        for k, v in enc_cfg.items():
            flat[f"encoder_{k}"] = v
    # dit_config → voxcpm.dit.*
    dit_cfg = cfg.get("dit_config", {})
    if dit_cfg:
        for k, v in dit_cfg.items():
            flat[f"dit_{k}"] = v
        cfm_cfg = dit_cfg.get("cfm_config", {})
        if cfm_cfg:
            for k, v in cfm_cfg.items():
                flat[f"cfm_{k}"] = v
    # audio_vae_config → voxcpm.vae.*
    vae_cfg = cfg.get("audio_vae_config", {})
    if vae_cfg:
        for k, v in vae_cfg.items():
            flat[f"vae_{k}"] = v

    return flat


def load_safetensors_index(hf_dir: Path) -> Dict[str, str]:
    """Return {tensor_name: file_path} mapping from safetensors."""
    index_path = hf_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r") as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", {})
        log.info("Found sharded safetensors index: %d tensors in %d files",
                 len(weight_map), len(set(weight_map.values())))
        return weight_map

    # Single file
    safetensors_files = sorted(hf_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files in {hf_dir}")
    log.info("Found %d safetensors file(s)", len(safetensors_files))

    weight_map = {}
    for sf_path in safetensors_files:
        with safe_open(sf_path, framework="pt") as sf:
            for key in sf.keys():
                weight_map[key] = sf_path.name
    return weight_map


def map_tensor_name(upstream_name: str) -> Optional[Tuple[str, bool]]:
    """Map upstream safetensors tensor name to canonical GGUF name.

    Returns (canonical_name, was_fallthrough) or None if the tensor
    should be skipped.
    """
    # Skip optimizer/excluded buffers
    if any(x in upstream_name for x in ("optimizer", "num_batches_tracked",
                                          "running_mean", "running_var")):
        return None

    # Track whether we hit a generic fallthrough (last entries)
    generic_start = len(PREFIX_MAP) - 5  # last 5 are generic fallthrough

    for idx, (pattern_re, template) in enumerate(PREFIX_MAP):
        m = re.match(pattern_re, upstream_name)
        if not m:
            continue

        # Build canonical name from groups
        groups = m.groups()
        if not groups:
            canonical = template
        else:
            parts = []
            # The template uses {} for each captured group
            template_parts = template.split("{}")
            g_idx = 0
            result = []
            for tp in template_parts:
                result.append(tp)
                if g_idx < len(groups):
                    result.append(str(groups[g_idx]))
                    g_idx += 1
            canonical = "".join(result)

        # Strip trailing dot
        canonical = canonical.rstrip(".")
        return (canonical, idx >= generic_start)

    # No pattern matched — log and use as-is
    log.warning("No mapping pattern for tensor: %s (using as-is)", upstream_name)
    return (upstream_name, True)


def value_from_config(cfg: dict, *keys: str, default=0):
    """Get first existing key from config.

    Note: keys may be float or int; preserve the original type instead
    of always truncating to int (bug fix: was ``int(v)`` which turns
    1e-05 → 0).
    """
    for k in keys:
        if k in cfg:
            v = cfg[k]
            if isinstance(v, float):
                return v
            if isinstance(v, int):
                return v
            return v
    return default


def write_gguf_metadata(writer: gguf.GGUFWriter, cfg: dict):
    """Write all GGUF metadata keys from HF config."""
    # General (architecture is already set by GGUFWriter constructor)
    writer.add_string("general.name", value_from_config(cfg, "_name_or_path", default="VoxCPM2"))
    writer.add_string("general.description",
                      "VoxCPM2 TTS model converted to GGUF")
    writer.add_int32("general.file_type", gguf.GGMLQuantizationType.F16.value)
    writer.add_string("general.converted_by", "voxcpm-c/convert_voxcpm2_to_gguf.py")

    # Audio dimensions (from top-level or audio_vae_config)
    writer.add_int32("voxcpm.patch_size", value_from_config(cfg, "patch_size"))
    writer.add_int32("voxcpm.feat_dim", value_from_config(cfg, "feat_dim"))
    writer.add_int32("voxcpm.latent_dim", value_from_config(cfg,
                      "scalar_quantization_latent_dim", "vae_latent_dim"))
    writer.add_int32("voxcpm.max_length", value_from_config(cfg, "max_length", "lm_max_position_embeddings"))
    writer.add_int32("voxcpm.sample_rate", value_from_config(cfg,
                      "vae_out_sample_rate", "vae_sample_rate"))
    writer.add_int32("voxcpm.encode_sample_rate", value_from_config(cfg, "vae_sample_rate"))

    # Special tokens
    writer.add_int32("voxcpm.audio_start_token", cfg.get("audio_start_token", 101))
    writer.add_int32("voxcpm.audio_end_token", cfg.get("audio_end_token", 102))
    writer.add_int32("voxcpm.ref_audio_start_token", cfg.get("ref_audio_start_token", 103))
    writer.add_int32("voxcpm.ref_audio_end_token", cfg.get("ref_audio_end_token", 104))

    # Flags
    writer.add_bool("voxcpm.supports_reference_audio",
                    cfg.get("supports_reference_audio", True))
    writer.add_bool("voxcpm.supports_streaming",
                    cfg.get("supports_streaming", True))

    # Base LM config (from lm_config or flattened lm_*)
    writer.add_int32("voxcpm.hidden_size", value_from_config(cfg, "lm_hidden_size"))
    writer.add_int32("voxcpm.num_hidden_layers", value_from_config(cfg, "lm_num_hidden_layers"))
    writer.add_int32("voxcpm.num_attention_heads", value_from_config(cfg, "lm_num_attention_heads"))
    writer.add_int32("voxcpm.num_kv_heads", value_from_config(cfg,
                      "lm_num_key_value_heads", "lm_num_kv_heads"))
    writer.add_int32("voxcpm.intermediate_size", value_from_config(cfg, "lm_intermediate_size"))
    writer.add_float32("voxcpm.rms_norm_eps", float(value_from_config(cfg,
                        "lm_rms_norm_eps", default=1e-5)))
    writer.add_int32("voxcpm.rope_theta", value_from_config(cfg, "lm_rope_theta"))
    writer.add_int32("voxcpm.max_seq_len", value_from_config(cfg,
                      "lm_max_position_embeddings", "max_length"))
    writer.add_float32("voxcpm.scale_depth",
                       float(cfg.get("lm_scale_depth", 0.0)))

    # Residual LM
    writer.add_int32("voxcpm.res_num_layers", value_from_config(cfg,
                      "residual_lm_num_layers"))
    writer.add_bool("voxcpm.res_no_rope",
                    bool(cfg.get("residual_lm_no_rope", False)))
    # RALM copies lm_config (including scale_depth) from model setup
    writer.add_float32("voxcpm.res_scale_depth",
                       float(cfg.get("lm_scale_depth", 0.0)))

    # Feat encoder (from encoder_config)
    writer.add_int32("voxcpm.enc_hidden_size", value_from_config(cfg, "encoder_hidden_dim"))
    writer.add_int32("voxcpm.enc_num_layers", value_from_config(cfg, "encoder_num_layers"))
    writer.add_int32("voxcpm.enc_num_heads", value_from_config(cfg, "encoder_num_heads"))

    # LocDiT (from dit_config)
    writer.add_int32("voxcpm.dit_hidden_size", value_from_config(cfg, "dit_hidden_dim"))
    writer.add_int32("voxcpm.dit_num_layers", value_from_config(cfg, "dit_num_layers"))
    writer.add_int32("voxcpm.dit_num_heads", value_from_config(cfg, "dit_num_heads"))

    # CFM solver
    writer.add_float32("voxcpm.cfm_sigma_min", float(value_from_config(cfg,
                        "cfm_sigma_min", default=1e-6)))
    writer.add_string("voxcpm.cfm_solver",
                      str(cfg.get("cfm_solver", "euler")))
    writer.add_float32("voxcpm.cfm_inference_cfg_rate",
                       float(value_from_config(cfg, "cfm_inference_cfg_rate", default=2.0)))

    # AudioVAE
    writer.add_int32("voxcpm.vae_encoder_dim", value_from_config(cfg, "vae_encoder_dim"))
    writer.add_int32("voxcpm.vae_latent_dim", value_from_config(cfg, "vae_latent_dim"))
    writer.add_int32("voxcpm.vae_decoder_dim", value_from_config(cfg, "vae_decoder_dim"))
    writer.add_int32("voxcpm.vae_sample_rate", value_from_config(cfg, "vae_sample_rate"))
    writer.add_int32("voxcpm.vae_out_sample_rate", value_from_config(cfg,
                      "vae_out_sample_rate", "vae_sample_rate"))

    # Tokenizer
    writer.add_int32("voxcpm.vocab_size", value_from_config(cfg, "lm_vocab_size", "vocab_size"))
    writer.add_int32("voxcpm.bos_token_id", value_from_config(cfg,
                      "lm_bos_token_id", "bos_token_id"))
    writer.add_int32("voxcpm.eos_token_id", value_from_config(cfg,
                      "lm_eos_token_id", "eos_token_id"))

    log.info("Wrote GGUF metadata")


def write_tokenizer(writer: gguf.GGUFWriter, hf_dir: Path, cfg: dict):
    """Write tokenizer metadata from HF snapshot."""
    tok_path = hf_dir / "tokenizer.json"
    if not tok_path.exists():
        log.warning("No tokenizer.json found; writing minimal tokenizer metadata")
        vocab_size = value_from_config(cfg, "lm_vocab_size", "vocab_size")
        if vocab_size:
            writer.add_string("tokenizer.ggml.model", "llama")
            writer.add_int32("tokenizer.ggml.bos_token_id",
                             cfg.get("lm_bos_token_id", cfg.get("bos_token_id", 1)))
            writer.add_int32("tokenizer.ggml.eos_token_id",
                             cfg.get("lm_eos_token_id", cfg.get("eos_token_id", 2)))
            tokens = [f"<token_{i}>" for i in range(vocab_size)]
            scores = [0.0] * vocab_size
            types = [gguf.TokenType.NORMAL] * vocab_size
            writer.add_token_list(tokens)
            writer.add_token_scores(scores)
            writer.add_token_types(types)
        return

    with open(tok_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    # Extract vocabulary
    vocab = tok_data.get("model", {}).get("vocab", {})
    if not vocab:
        vocab = {}
        for t in tok_data.get("added_tokens", []):
            vocab[t["content"]] = t["id"]

    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
    tokens = [t[0] for t in sorted_tokens]
    scores = [0.0] * len(tokens)
    types = []

    added_vocab = {t["content"]: t for t in tok_data.get("added_tokens", [])}
    special_ids = {
        cfg.get("lm_bos_token_id", cfg.get("bos_token_id", 1)),
        cfg.get("lm_eos_token_id", cfg.get("eos_token_id", 2)),
        cfg.get("audio_start_token", 101),
        cfg.get("audio_end_token", 102),
        cfg.get("ref_audio_start_token", 103),
        cfg.get("ref_audio_end_token", 104),
    }

    for token_id, (tok_str, _) in enumerate(sorted_tokens):
        if tok_str in added_vocab or token_id in special_ids:
            types.append(gguf.TokenType.CONTROL)
        else:
            types.append(gguf.TokenType.NORMAL)

    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    merges = tok_data.get("model", {}).get("merges", [])
    if merges:
        writer.add_token_merges(merges)
    writer.add_string("tokenizer.ggml.model", "llama")
    writer.add_int32("tokenizer.ggml.bos_token_id",
                     cfg.get("lm_bos_token_id", cfg.get("bos_token_id", 1)))
    writer.add_int32("tokenizer.ggml.eos_token_id",
                     cfg.get("lm_eos_token_id", cfg.get("eos_token_id", 2)))

    log.info("Wrote tokenizer: %d tokens", len(tokens))


def convert_to_dtype(arr: np.ndarray, canonical_name: str, outtype: str) -> np.ndarray:
    """Convert numpy array to target dtype."""
    # Norms and biases are always kept in f32
    is_norm = "norm" in canonical_name or "layernorm" in canonical_name
    is_bias = canonical_name.endswith(".bias")

    if is_norm or is_bias or outtype == "f32":
        return arr.astype(np.float32)
    elif outtype == "f16":
        return arr.astype(np.float16)
    elif outtype == "bf16":
        # numpy doesn't have native bf16; store as f32, the gguf C runtime
        # will handle bf16 conversion at load time
        return arr.astype(np.float32)
    elif outtype.startswith("q"):
        return arr.astype(np.float32)
    else:
        log.warning("Unknown outtype %s; keeping as f32", outtype)
        return arr.astype(np.float32)


def add_tensor_to_writer(writer: gguf.GGUFWriter,
                         canonical_name: str,
                         arr: np.ndarray,
                         outtype: str,
                         tensor_dtype_map: Dict[str, str]):
    """Add tensor to GGUF writer with appropriate dtype/quantization."""
    is_norm = "norm" in canonical_name or "layernorm" in canonical_name
    is_bias = canonical_name.endswith(".bias")

    if is_norm or is_bias or outtype == "f32":
        actual_type = gguf.GGMLQuantizationType.F32
        data = arr.astype(np.float32) if arr.dtype != np.float32 else arr
    elif outtype == "f16":
        actual_type = gguf.GGMLQuantizationType.F16
        data = arr.astype(np.float16) if arr.dtype not in (np.float16,) else arr
    elif outtype == "bf16":
        actual_type = gguf.GGMLQuantizationType.F32
        data = arr.astype(np.float32)
    elif outtype.startswith("q"):
        actual_type = gguf.GGMLQuantizationType.F32
        data = arr.astype(np.float32)
    else:
        actual_type = gguf.GGMLQuantizationType.F32
        data = arr.astype(np.float32)

    writer.add_tensor(canonical_name, data, raw_dtype=actual_type)
    tensor_dtype_map[canonical_name] = actual_type.name


def load_audiovae_pth(pth_path: Path) -> Dict[str, torch.Tensor]:
    """Load audiovae.pth, convert weight_norm → plain weight, canonicalize names."""
    log.info("Loading audiovae.pth: %s", pth_path)
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("state_dict", ckpt)
    log.info("  Loaded %d tensors from audiovae.pth", len(sd))

    # Collect key groups: {base_key: {g_or_v: tensor}}
    # weight_norm pairs: {base}.weight_g + {base}.weight_v → {base}.weight
    norm_groups: Dict[str, Dict[str, torch.Tensor]] = {}
    plain_tensors: Dict[str, torch.Tensor] = {}
    for key, tensor in sd.items():
        if key.endswith(".weight_g"):
            base = key[:-len("_g")]
            norm_groups.setdefault(base, {})["g"] = tensor
        elif key.endswith(".weight_v"):
            base = key[:-len("_v")]
            norm_groups.setdefault(base, {})["v"] = tensor
        else:
            plain_tensors[key] = tensor

    # Resolve weight_norm: weight = g * normalize(v)
    resolved = dict(plain_tensors)
    for base_key, gv in norm_groups.items():
        if "g" in gv and "v" in gv:
            g = gv["g"]  # [OC, 1, 1]
            v = gv["v"]  # [OC, IC, K]
            # Normalize v along dims (1, 2) for Conv1d
            norm_v = torch.norm(v, dim=(1, 2), keepdim=True)
            norm_v = torch.clamp(norm_v, min=1e-8)
            weight = g * v / norm_v
            resolved[base_key + ".weight"] = weight
            log.debug("  Resolved weight_norm: %s → %s", base_key, weight.shape)
        else:
            log.warning("  Incomplete weight_norm pair: %s", base_key)

    log.info("  Resolved %d weight_norm pairs → plain weight", len(norm_groups))
    return resolved


def canonicalize_audiovae_name(upstream_name: str) -> str:
    """Map audiovae.pth tensor name to canonical GGUF name (audio_vae. prefix)."""
    # Already prefixed with encoder. or decoder.
    # Just add audio_vae. prefix
    if upstream_name.startswith("encoder.") or upstream_name.startswith("decoder."):
        return f"audio_vae.{upstream_name}"
    return f"audio_vae.{upstream_name}"


def write_shape_manifest(mapping: Dict[str, Dict], out_path: str):
    """Write shape manifest JSON."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    log.info("Wrote shape manifest: %s (%d entries)", out_path, len(mapping))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert VoxCPM2 HF snapshot to GGUF")
    parser.add_argument("--hf-dir", required=True,
                        help="Path to Hugging Face VoxCPM2 snapshot")
    parser.add_argument("--out", required=True,
                        help="Output GGUF path (.gguf)")
    parser.add_argument("--outtype", default="f16",
                        choices=["f32", "f16", "bf16", "q8_0", "q4_k_m", "q5_k_m"],
                        help="Output quantization type (default: f16)")
    parser.add_argument("--emit-shape-manifest", default=None,
                        help="Optional path for shapes.json manifest")
    parser.add_argument("--dry-run", action="store_true",
                        help="List tensor mappings without writing GGUF")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    parser.add_argument("--audiovae-pth", default=None,
                        help="Path to audiovae.pth (optional)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    hf_dir = Path(args.hf_dir)
    if not hf_dir.is_dir():
        log.error("HF dir not found: %s", hf_dir)
        return 1

    # ---- Load config ----
    cfg = load_config(hf_dir)

    # ---- Load safetensors index ----
    try:
        weight_map = load_safetensors_index(hf_dir)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1

    # ---- Map tensor names ----
    mapped: Dict[str, Dict] = {}  # canonical_name → {source, source_shape, gguf_shape, dtype}
    unmapped: List[str] = []

    for upstream_name, file_name in weight_map.items():
        result = map_tensor_name(upstream_name)
        if result is None:
            continue
        canonical, was_fallthrough = result
        if was_fallthrough:
            unmapped.append(upstream_name)

        mapped[canonical] = {
            "source_name": upstream_name,
            "source_file": file_name,
            "source_shape": [],
            "gguf_shape": [],
            "dtype": "",
            "quantized": args.outtype.startswith("q"),
        }

    if unmapped:
        log.warning("%d tensors have no explicit mapping pattern:", len(unmapped))
        for name in unmapped[:10]:
            log.warning("  %s", name)
        if len(unmapped) > 10:
            log.warning("  ... and %d more", len(unmapped) - 10)

    log.info("Mapped %d safetensors (of %d total)",
             len(mapped), len(weight_map))

    # ---- Load audiovae.pth (optional, auto-detected in standard HF snapshot) ----
    audiovae_tensors: Dict[str, torch.Tensor] = {}
    av_path = Path(args.audiovae_pth) if args.audiovae_pth else (hf_dir / "audiovae.pth")
    if av_path.exists():
        av_raw = load_audiovae_pth(av_path)
        for upstream_name, tensor in av_raw.items():
            canonical = canonicalize_audiovae_name(upstream_name)
            # Convert torch → numpy (bfloat16 → float32 first)
            if hasattr(tensor, 'dtype') and tensor.dtype == torch.bfloat16:
                arr = tensor.cpu().float().numpy()
            else:
                arr = tensor.cpu().numpy()
            audiovae_tensors[canonical] = arr
            if canonical not in mapped:
                mapped[canonical] = {
                    "source_name": upstream_name,
                    "source_file": "audiovae.pth",
                    "source_shape": list(arr.shape),
                    "gguf_shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "quantized": False,
                }
        log.info("Loaded %d tensors from audiovae.pth", len(audiovae_tensors))
    elif args.audiovae_pth:
        log.warning("audiovae.pth not found: %s", av_path)
    else:
        log.warning("No audiovae.pth found under HF dir; generated GGUF will not support TTS decode")

    # ---- Dry run ----
    if args.dry_run:
        print(f"\n=== Dry Run: {args.hf_dir} → {args.out} ===")
        print(f"Output type: {args.outtype}")
        print(f"Total safetensors: {len(weight_map)}")
        print(f"Mapped tensors (safetensors): {len(mapped)}")
        if audiovae_tensors:
            print(f"AudioVAE tensors: {len(audiovae_tensors)}")
        if unmapped:
            print(f"\nUnmapped tensors ({len(unmapped)}):")
            for name in unmapped[:10]:
                print(f"  {name}")
        print("\nCanonical tensor names:")
        for cname in sorted(mapped.keys()):
            info = mapped[cname]
            print(f"  {cname}  ← {info['source_name']}")
        print("\nDry run complete. No file written.")
        return 0

    # ---- Write GGUF ----
    log.info("Writing GGUF: %s (outtype=%s)", args.out, args.outtype)
    writer = gguf.GGUFWriter(args.out, "voxcpm2")

    # Metadata
    write_gguf_metadata(writer, cfg)
    write_tokenizer(writer, hf_dir, cfg)

    # Read safetensors and write tensors
    tensor_dtype_map: Dict[str, str] = {}
    n_written = 0
    n_skipped = 0
    read_files: Dict[str, Any] = {}

    def sort_key(cname: str) -> Tuple:
        """Sort: base_lm first, then modules in expected order."""
        if cname.startswith("base_lm"):
            return (0, cname)
        elif cname.startswith("residual_lm"):
            return (1, cname)
        elif cname.startswith("feat_encoder"):
            return (2, cname)
        elif cname.startswith("feat_decoder"):
            return (3, cname)
        elif cname.startswith("fsq"):
            return (4, cname)
        elif cname.startswith("enc_to_lm"):
            return (5, cname)
        elif cname.startswith("lm_to_dit"):
            return (6, cname)
        elif cname.startswith("res_to_dit"):
            return (7, cname)
        elif cname.startswith("fusion"):
            return (8, cname)
        elif cname.startswith("stop_"):
            return (9, cname)
        elif cname.startswith("audio_vae"):
            return (10, cname)
        else:
            return (99, cname)

    # Write safetensors first, then audiovae
    for canonical_name in sorted(mapped.keys(), key=sort_key):
        info = mapped[canonical_name]

        # Skip audiovae — handled separately
        if info["source_file"] == "audiovae.pth":
            continue

        source_name = info["source_name"]
        source_file = info["source_file"]

        if source_file not in read_files:
            sf_path = hf_dir / source_file
            if not sf_path.exists():
                log.error("Missing safetensors file: %s", sf_path)
                n_skipped += 1
                continue
            try:
                # Use PyTorch backend for bfloat16 support
                read_files[source_file] = safe_open(str(sf_path), framework="pt")
            except Exception as e:
                log.error("Cannot open %s: %s", sf_path, e)
                n_skipped += 1
                continue

        sf = read_files[source_file]
        if source_name not in sf.keys():
            log.warning("Tensor %s not found in %s (keys=%d)",
                        source_name, source_file, len(sf.keys()))
            n_skipped += 1
            continue

        try:
            tensor = sf.get_tensor(source_name)  # returns torch.Tensor
        except Exception as e:
            log.error("Error reading %s: %s", source_name, e)
            n_skipped += 1
            continue

        # Convert to numpy (bfloat16 → float32 first)
        if tensor.dtype == torch.bfloat16:
            arr = tensor.cpu().float().numpy()
        else:
            arr = tensor.cpu().numpy()

        # Update shape info
        info["source_shape"] = list(arr.shape)
        info["dtype"] = str(arr.dtype)

        # GGUF tensor layout: ne[0]=innermost dim
        if arr.ndim == 2:
            info["gguf_shape"] = [arr.shape[1], arr.shape[0]]
        elif arr.ndim == 1:
            info["gguf_shape"] = [arr.shape[0]]
        else:
            info["gguf_shape"] = list(arr.shape)

        # Convert dtype
        data = convert_to_dtype(arr, canonical_name, args.outtype)

        # Add to GGUF
        add_tensor_to_writer(writer, canonical_name, data, args.outtype,
                             tensor_dtype_map)
        n_written += 1

        if n_written % 50 == 0:
            log.info("  Wrote %d/%d tensors...", n_written, len(mapped))

    # Close safetensors files (safe_open is a context manager, not a file handle)
    read_files.clear()

    # Write audiovae tensors
    for canonical_name in sorted(audiovae_tensors.keys()):
        arr = audiovae_tensors[canonical_name]
        data = convert_to_dtype(arr, canonical_name, args.outtype)
        add_tensor_to_writer(writer, canonical_name, data, args.outtype,
                             tensor_dtype_map)
        n_written += 1
        log.info("  Wrote audiovae tensor: %s [%s]", canonical_name, list(arr.shape))

    # Write GGUF file
    log.info("Writing %d tensors to %s...", n_written, args.out)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    log.info("Done: %s (%d tensors written, %d skipped)",
             args.out, n_written, n_skipped)

    # ---- Emit shape manifest ----
    if args.emit_shape_manifest:
        write_shape_manifest(mapped, args.emit_shape_manifest)

    return 0 if n_written > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
