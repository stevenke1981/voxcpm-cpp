#!/usr/bin/env python3
"""
VoxCPM2 Reference Fixture Exporter.

Loads the upstream VoxCPM2 Python model, runs a short zero-shot generation,
and dumps intermediate tensors at every pipeline stage as .npy files.

Usage:
    python tools/export_ref_fixtures.py [--model-dir model_download] [--text "Hello world."] [--out-dir fixtures/ref]

Output:
    fixtures/ref/*.npy  — intermediate tensors at each stage
    fixtures/ref/manifest.json  — metadata about each tensor
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

# Ensure we use the installed voxcpm package
from voxcpm import VoxCPM
from voxcpm.model.voxcpm2 import VoxCPM2Model

log = logging.getLogger("export_ref_fixtures")

# ---------------------------------------------------------------------------
# Monkey-patch VoxCPM2Model._inference to dump intermediates
# ---------------------------------------------------------------------------

_ORIG_INFERENCE = None
_FIXTURE_DIR = None
_STEP_COUNTER = [0]


def _dump_tensor(name: str, tensor: torch.Tensor, step: int = -1):
    """Save tensor to .npy file and record in manifest."""
    global _FIXTURE_DIR
    if _FIXTURE_DIR is None:
        return
    arr = tensor.detach().cpu().float().numpy()
    if step >= 0:
        fname = f"step{step:04d}_{name}.npy"
    else:
        fname = f"{name}.npy"
    path = os.path.join(_FIXTURE_DIR, fname)
    np.save(path, arr)
    # Log shape info
    log.info("  Dumped %s: %s → %s", fname, list(arr.shape), path)


def _patched_inference(
    self,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    feat: torch.Tensor,
    feat_mask: torch.Tensor,
    min_len: int = 2,
    max_len: int = 2000,
    inference_timesteps: int = 10,
    cfg_value: float = 2.0,
    streaming: bool = False,
    streaming_prefix_len: int = 4,
):
    """Patched _inference that dumps intermediates."""
    B, T, P, D = feat.shape
    log.info("=== _inference (patched) ===")
    log.info("Input shapes: text=%s text_mask=%s feat=%s feat_mask=%s",
             list(text.shape), list(text_mask.shape), list(feat.shape), list(feat_mask.shape))

    # Dump inputs
    _dump_tensor("input_text", text)
    _dump_tensor("input_text_mask", text_mask)
    _dump_tensor("input_feat", feat)
    _dump_tensor("input_feat_mask", feat_mask)

    # ---- Prompt Eval: Feature Encoder ----
    prefill_encoder = getattr(self, "_feat_encoder_raw", self.feat_encoder)
    feat_embed = prefill_encoder(feat)  # [b, t, h_feat]
    _dump_tensor("feat_encoder_out", feat_embed)
    log.info("feat_encoder out: %s", list(feat_embed.shape))

    feat_embed = self.enc_to_lm_proj(feat_embed)
    _dump_tensor("enc_to_lm_proj_out", feat_embed)
    log.info("enc_to_lm_proj out: %s", list(feat_embed.shape))

    # ---- Text Embedding ----
    scale_emb = 1.0
    if self.config.lm_config.use_mup:
        scale_emb = self.config.lm_config.scale_emb
    text_embed = self.base_lm.embed_tokens(text) * scale_emb
    _dump_tensor("text_embed", text_embed)
    log.info("text_embed: %s", list(text_embed.shape))

    # ---- Combined Embedding ----
    combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed
    _dump_tensor("combined_embed", combined_embed)
    log.info("combined_embed: %s", list(combined_embed.shape))

    # ---- Base LM Prompt Eval ----
    enc_outputs, kv_cache_tuple = self.base_lm(
        inputs_embeds=combined_embed,
        is_causal=True,
    )
    self.base_lm.kv_cache.fill_caches(kv_cache_tuple)
    _dump_tensor("base_lm_out", enc_outputs)
    log.info("base_lm out: %s", list(enc_outputs.shape))

    # ---- FSQ ----
    enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
    _dump_tensor("fsq_out", enc_outputs)
    log.info("fsq out: %s", list(enc_outputs.shape))

    lm_hidden = enc_outputs[:, -1, :]
    _dump_tensor("lm_hidden_init", lm_hidden)
    log.info("lm_hidden init: %s", list(lm_hidden.shape))

    # ---- Residual LM Prompt Eval ----
    residual_enc_inputs = self.fusion_concat_proj(
        torch.cat((enc_outputs, feat_mask.unsqueeze(-1) * feat_embed), dim=-1)
    )
    _dump_tensor("residual_enc_inputs", residual_enc_inputs)
    log.info("residual_enc_inputs: %s", list(residual_enc_inputs.shape))

    residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(
        inputs_embeds=residual_enc_inputs,
        is_causal=True,
    )
    self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
    _dump_tensor("residual_lm_out", residual_enc_outputs)
    log.info("residual_lm out: %s", list(residual_enc_outputs.shape))

    residual_hidden = residual_enc_outputs[:, -1, :]
    _dump_tensor("residual_hidden_init", residual_hidden)
    log.info("residual_hidden init: %s", list(residual_hidden.shape))

    # ---- Initial DiT hidden ----
    dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)
    dit_hidden_2 = self.res_to_dit_proj(residual_hidden)
    dit_hidden = torch.cat((dit_hidden_1, dit_hidden_2), dim=-1)
    _dump_tensor("dit_hidden_init", dit_hidden)
    log.info("dit_hidden init: %s", list(dit_hidden.shape))

    # ---- Autoregressive Loop ----
    prefix_feat_cond = feat[:, -1, ...]  # b, p, d
    pred_feat_seq = []
    has_continuation_audio = feat_mask[0, -1].item() == 1
    context_len = 0
    if has_continuation_audio:
        audio_indices = feat_mask.squeeze(0).nonzero(as_tuple=True)[0]
        context_len = min(streaming_prefix_len - 1, len(audio_indices))
        last_audio_indices = audio_indices[-context_len:]
        pred_feat_seq = list(feat[:, last_audio_indices, :, :].split(1, dim=1))

    _STEP_COUNTER[0] = 0
    for i in range(max_len):
        step = _STEP_COUNTER[0]
        log.info("--- Autoregressive step %d/%d ---", i, max_len)

        # DiT hidden for this step
        dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)
        dit_hidden_2 = self.res_to_dit_proj(residual_hidden)
        dit_hidden = torch.cat((dit_hidden_1, dit_hidden_2), dim=-1)
        _dump_tensor("dit_hidden", dit_hidden, step)
        _dump_tensor("lm_to_dit_proj_out", dit_hidden_1, step)
        _dump_tensor("res_to_dit_proj_out", dit_hidden_2, step)

        # CFM / Diffusion
        cond_input = prefix_feat_cond.transpose(1, 2).contiguous()
        _dump_tensor("cfm_cond", cond_input, step)

        # Patch the CFM decoder to dump intermediates
        pred_feat = self.feat_decoder(
            mu=dit_hidden,
            patch_size=self.patch_size,
            cond=cond_input,
            n_timesteps=inference_timesteps,
            cfg_value=cfg_value,
        ).transpose(1, 2)  # [b, p, d]

        _dump_tensor("cfm_pred_feat", pred_feat, step)
        log.info("  cfm pred_feat: %s range=[%.4f, %.4f]",
                 list(pred_feat.shape), pred_feat.min().item(), pred_feat.max().item())

        # Encode predicted feature for next step
        curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))  # b, 1, c
        _dump_tensor("curr_embed_raw", curr_embed, step)
        curr_embed = self.enc_to_lm_proj(curr_embed)
        _dump_tensor("curr_embed_proj", curr_embed, step)

        pred_feat_seq.append(pred_feat.unsqueeze(1))

        # Save prefix_feat_cond before overwriting
        _dump_tensor("prefix_feat_cond", prefix_feat_cond, step)
        prefix_feat_cond = pred_feat

        # Stop prediction
        stop_hidden = self.stop_actn(self.stop_proj(lm_hidden))
        _dump_tensor("stop_hidden", stop_hidden, step)
        stop_logits = self.stop_head(stop_hidden)
        _dump_tensor("stop_logits", stop_logits, step)
        stop_flag = stop_logits.argmax(dim=-1)[0].cpu().item()

        log.info("  stop_logits=[%.4f, %.4f] argmax=%d",
                 stop_logits[0, 0].item(), stop_logits[0, 1].item(), stop_flag)

        if i > min_len and stop_flag == 1:
            log.info("  STOP at step %d", i)
            break

        # ---- Next LM step ----
        lm_hidden = self.base_lm.forward_step(
            curr_embed[:, 0, :],
            torch.tensor([self.base_lm.kv_cache.step()], device=curr_embed.device),
        ).clone()
        _dump_tensor("lm_hidden_step", lm_hidden, step)

        lm_hidden = self.fsq_layer(lm_hidden)
        _dump_tensor("lm_hidden_fsq", lm_hidden, step)

        curr_residual_input = self.fusion_concat_proj(
            torch.cat((lm_hidden, curr_embed[:, 0, :]), dim=-1)
        )
        _dump_tensor("curr_residual_input", curr_residual_input, step)

        residual_hidden = self.residual_lm.forward_step(
            curr_residual_input,
            torch.tensor([self.residual_lm.kv_cache.step()], device=curr_embed.device),
        ).clone()
        _dump_tensor("residual_hidden_step", residual_hidden, step)

        _STEP_COUNTER[0] += 1

    # ---- Final output ----
    pred_feat_seq_tensor = torch.cat(pred_feat_seq, dim=1)  # b, t, p, d
    _dump_tensor("pred_feat_seq_full", pred_feat_seq_tensor)

    # Flatten to [b, d, t*p] for VAE decoder (same as original _generate)
    B_actual = pred_feat_seq_tensor.shape[0]
    feat_pred = rearrange(pred_feat_seq_tensor, "b t p d -> b d (t p)", b=B_actual, p=self.patch_size)
    _dump_tensor("feat_pred_latent", feat_pred)
    log.info("feat_pred latent shape: %s", list(feat_pred.shape))

    # VAE decode
    decode_audio = self.audio_vae.decode(feat_pred.to(torch.float32))
    _dump_tensor("vae_decode_raw", decode_audio)
    log.info("vae_decode_raw shape: %s", list(decode_audio.shape))

    generated_feat = pred_feat_seq_tensor[:, context_len:, :, :].squeeze(0).cpu()
    _dump_tensor("generated_feat", generated_feat)

    if not streaming:
        pred_feat_seq_tensor = torch.cat(pred_feat_seq, dim=1)
        B_actual = pred_feat_seq_tensor.shape[0]
        feat_pred = rearrange(pred_feat_seq_tensor, "b t p d -> b d (t p)", b=B_actual, p=self.patch_size)
        generated_feat = pred_feat_seq_tensor[:, context_len:, :, :].squeeze(0).cpu()
        yield feat_pred, generated_feat, context_len


# ---------------------------------------------------------------------------
# Also patch feat_decoder (UnifiedCFM) to dump diffusion trajectory
# ---------------------------------------------------------------------------

_ORIG_CFM_FORWARD = None


def _make_cfm_patch(orig_bound_forward):
    """Create a patched CFM decorator that dumps output.
    
    orig_bound_forward is already bound to the feat_decoder instance.
    Original forward signature: forward(self, mu, n_timesteps, patch_size, cond, ...)
    """
    def _patched_cfm_forward(
        self,
        mu: torch.Tensor,
        n_timesteps: int,
        patch_size: int,
        cond: torch.Tensor,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        sway_sampling_coef: float = 1.0,
        use_cfg_zero_star: bool = True,
    ):
        log.info("  === CFM forward: mu=%s n_steps=%d psize=%d cond=%s cfg=%.2f ===",
                 list(mu.shape), n_timesteps, patch_size, list(cond.shape), cfg_value)
        # Use keyword args to avoid order mismatch
        result = orig_bound_forward(
            mu=mu, n_timesteps=n_timesteps, patch_size=patch_size,
            cond=cond, temperature=temperature, cfg_value=cfg_value,
            sway_sampling_coef=sway_sampling_coef, use_cfg_zero_star=use_cfg_zero_star,
        )
        _dump_tensor("cfm_final_out", result)
        log.info("  cfm_final_out: %s range=[%.6f, %.6f]",
                 list(result.shape), result.min().item(), result.max().item())
        return result
    return _patched_cfm_forward


# ---------------------------------------------------------------------------
# Also patch feat_decoder's internal _sample (CFM solver) to dump step trajectory
# ---------------------------------------------------------------------------


def _patch_cfm_sampler(cfm_module):
    """Patch CFM sampler to dump intermediate diffusion trajectory.

    Uses compound naming 'ar{ARstep}_d{Dstep}_cfm_traj_state' to avoid
    collision between autoregressive step counters and diffusion step indices.

    Dumps:
      - ar0000_d0000_cfm_traj_state.npy  (initial noise x_1 for AR step 0)
      - ar0000_d0001_cfm_traj_state.npy (intermediate state)
      - ar0000_cfm_noise.npy            (initial noise, convenience alias)
      - ar0000_cfm_clean.npy            (final denoised, convenience alias)
      ...
    """
    if hasattr(cfm_module, '_sample') and not hasattr(cfm_module, '_sample_orig'):
        cfm_module._sample_orig = cfm_module._sample

        def _patched_sample(mu, patch_size, cond, n_timesteps, cfg_value):
            ar_step = _STEP_COUNTER[0]
            log.info("    CFM _sample (AR step %d): mu=%s n_steps=%d",
                     ar_step, list(mu.shape), n_timesteps)
            traj = cfm_module._sample_orig(mu, patch_size, cond, n_timesteps, cfg_value)
            # traj shape: [n_steps+1, batch*patches, d]
            # traj[0] = initial noise (x_1), traj[-1] = clean
            if isinstance(traj, torch.Tensor):
                for s in range(traj.shape[0]):
                    compound_name = f"ar{ar_step:04d}_d{s:04d}_cfm_traj_state"
                    _dump_tensor(compound_name, traj[s:s+1])
                # Convenience aliases
                _dump_tensor(f"ar{ar_step:04d}_cfm_noise", traj[0:1])
                _dump_tensor(f"ar{ar_step:04d}_cfm_clean", traj[-1:])
                log.info("    CFM trajectory (AR step %d): %s steps, %s -> noise=%s clean=%s",
                         ar_step, traj.shape[0], list(traj.shape),
                         list(traj[0].shape), list(traj[-1].shape))
            return traj

        cfm_module._sample = _patched_sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Export VoxCPM2 reference fixtures")
    parser.add_argument("--model-dir", default="model_download",
                        help="Path to VoxCPM2 HF snapshot directory")
    parser.add_argument("--text", default="Hello world.",
                        help="Text to generate")
    parser.add_argument("--out-dir", default="fixtures/ref",
                        help="Output directory for .npy files")
    parser.add_argument("--device", default=None,
                        help="Device (cpu, cuda, auto). Default: auto")
    parser.add_argument("--steps", type=int, default=10,
                        help="CFM inference timesteps")
    parser.add_argument("--cfg", type=float, default=2.0,
                        help="CFG value")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        log.error("Model directory not found: %s", model_dir)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global _FIXTURE_DIR
    _FIXTURE_DIR = str(out_dir)

    # ---- Load model ----
    log.info("Loading model from: %s", model_dir)
    log.info("Device: %s, optimize=False, dtype preserved", args.device or "auto")

    # Use VoxCPM2Model directly for lower-level control
    from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
    import json as _json

    with open(os.path.join(str(model_dir), "config.json"), "r", encoding="utf-8") as f:
        config_dict = _json.load(f)

    # We need to import AudioVAEV2 for V2
    from voxcpm.modules.audiovae import AudioVAEV2, AudioVAEConfigV2

    # Build VoxCPMConfig
    config = VoxCPMConfig.model_validate_json(_json.dumps(config_dict))

    # Load tokenizer
    from transformers import LlamaTokenizerFast
    tokenizer = LlamaTokenizerFast.from_pretrained(str(model_dir))

    # Build audio VAE
    audio_vae_config = getattr(config, "audio_vae_config", None)
    audio_vae = AudioVAEV2(config=audio_vae_config) if audio_vae_config else AudioVAEV2()

    # Determine device
    device = args.device
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    log.info("Using device: %s", device)

    # Create model
    model = VoxCPM2Model(config, tokenizer, audio_vae, lora_config=None, device=device)

    # Load weights
    safetensors_path = os.path.join(str(model_dir), "model.safetensors")
    audiovae_pth_path = os.path.join(str(model_dir), "audiovae.pth")

    from safetensors.torch import load_file as st_load
    log.info("Loading model.safetensors...")
    model_state_dict = st_load(safetensors_path, device="cpu")

    vae_ckpt = torch.load(audiovae_pth_path, map_location="cpu", weights_only=True)
    vae_state_dict = vae_ckpt.get("state_dict", vae_ckpt)
    for kw, val in vae_state_dict.items():
        model_state_dict[f"audio_vae.{kw}"] = val

    model.load_state_dict(model_state_dict, strict=False)

    # Convert to configured dtype (as in VoxCPM2Model.from_local)
    from voxcpm.model.utils import get_dtype
    lm_dtype = get_dtype(config.dtype)
    log.info("Converting model to dtype: %s (device: %s)", lm_dtype, device)
    model = model.to(lm_dtype)
    model = model.to(device).eval()
    # Move VAE to f32 (as in from_local)
    model.audio_vae = model.audio_vae.to(torch.float32)
    log.info("Model loaded successfully")

    # Save model config as reference
    config_json = config.model_dump(mode="json")
    with open(os.path.join(str(out_dir), "config.json"), "w") as f:
        json.dump(config_json, f, indent=2)
    log.info("Saved config to config.json")

    # ---- Patch inference for fixture dumping ----
    global _ORIG_INFERENCE
    _ORIG_INFERENCE = VoxCPM2Model._inference
    VoxCPM2Model._inference = _patched_inference

    # ---- Patch CFM forward to dump output ----
    cfm_patched = _make_cfm_patch(model.feat_decoder.forward)
    model.feat_decoder.forward = cfm_patched.__get__(model.feat_decoder, type(model.feat_decoder))

    # ---- Patch CFM sampler for trajectory ----
    _patch_cfm_sampler(model.feat_decoder)

    # ---- Dump reference config values ----
    ref_cfg = {
        "patch_size": model.patch_size,
        "feat_dim": model.feat_dim,
        "hidden_size": config.lm_config.hidden_size,
        "dit_hidden_size": config.dit_config.hidden_dim,
        "vae_latent_dim": audio_vae.latent_dim,
        "vae_sample_rate": audio_vae.sample_rate,
        "vae_out_sample_rate": getattr(audio_vae, "out_sample_rate", audio_vae.sample_rate),
        "chunk_size": audio_vae.chunk_size,
    }
    log.info("Reference config: %s", ref_cfg)

    # ---- Generate reference fixture ----
    log.info("Generating text: '%s'", args.text)

    # Use the _generate_with_prompt_cache path for consistency
    result = model.generate(
        target_text=args.text,
        min_len=2,
        max_len=512,
        inference_timesteps=args.steps,
        cfg_value=args.cfg,
    )
    audio = result.squeeze(0).cpu().numpy()
    log.info("Generated audio: %s samples, range=[%.4f, %.4f], RMS=%.4f",
             len(audio), audio.min(), audio.max(), np.sqrt(np.mean(audio ** 2)))

    # Save audio
    import soundfile as sf
    sf_path = os.path.join(str(out_dir), "ref_audio.wav")
    sf.write(sf_path, audio, ref_cfg["vae_out_sample_rate"])
    log.info("Saved reference audio: %s", sf_path)

    # ---- Build manifest ----
    # List all .npy files and build manifest
    manifest = {}
    for fpath in sorted(out_dir.glob("*.npy")):
        arr = np.load(str(fpath))
        manifest[fpath.name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size_bytes": int(arr.nbytes),
            "path": str(fpath.relative_to(out_dir.parent)),
        }

    manifest["_config"] = ref_cfg
    manifest["_generation_params"] = {
        "text": args.text,
        "steps": args.steps,
        "cfg": args.cfg,
        "device": device,
    }

    manifest_path = os.path.join(str(out_dir), "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Wrote manifest: %s (%d entries)", manifest_path, len(manifest))

    # ---- Summary ----
    npy_count = sum(1 for f in out_dir.glob("*.npy"))
    log.info("")
    log.info("=== Fixture Export Complete ===")
    log.info("Output directory: %s", out_dir)
    log.info("NPY files: %d", npy_count)
    log.info("Audio: %s (%.2f seconds at %d Hz)",
             sf_path, len(audio) / ref_cfg["vae_out_sample_rate"], ref_cfg["vae_out_sample_rate"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
