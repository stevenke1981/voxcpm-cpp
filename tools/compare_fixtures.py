#!/usr/bin/env python3
"""
C/Python Fixture Comparison Tool.

Loads Python reference fixtures (.npy files) and compares them against
C-side extracted values (from debug output or .npy files).

Usage:
    # Quick analyze Python reference
    python tools/compare_fixtures.py fixtures/ref --analyze

    # Compare C-extracted .npy files against Python reference
    python tools/compare_fixtures.py fixtures/ref --c-dir fixtures/c_dump
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("compare_fixtures")


def analyze_reference(ref_dir: str):
    """Analyze Python reference fixtures and print summary statistics."""
    ref_path = Path(ref_dir)
    if not ref_path.is_dir():
        log.error("Reference directory not found: %s", ref_dir)
        return

    npy_files = sorted(ref_path.glob("*.npy"))
    if not npy_files:
        log.error("No .npy files found in %s", ref_dir)
        return

    log.info("=== Reference Fixture Analysis ===")
    log.info("Directory: %s", ref_dir)
    log.info("Total .npy files: %d", len(npy_files))

    # Load manifest if exists
    manifest_path = ref_path / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        # Print generation params
        gen_params = manifest.get("_generation_params", {})
        if gen_params:
            log.info("Generation params: %s", gen_params)
        ref_cfg = manifest.get("_config", {})
        if ref_cfg:
            log.info("Model config: %s", ref_cfg)

    # Group tensors by pipeline stage
    stages = {
        "01-inputs": lambda n: n.startswith("input_"),
        "02-encoder": lambda n: n.startswith("feat_encoder") or n.startswith("enc_to_lm"),
        "03-embedding": lambda n: n.startswith("text_embed") or n.startswith("combined_embed"),
        "04-base_lm": lambda n: n.startswith("base_lm_out") or n.startswith("fsq_out") or n.startswith("lm_hidden_init"),
        "05-residual": lambda n: n.startswith("residual") or n.startswith("dit_hidden_init"),
        "06-step_cfm": lambda n: "cfm_pred_feat" in n or "cfm_final_out" in n,
        "07-step_lm": lambda n: "lm_hidden_step" in n or "lm_hidden_fsq" in n,
        "08-step_res": lambda n: "residual_hidden_step" in n or "curr_residual_input" in n,
        "09-step_stop": lambda n: "stop_logits" in n or "stop_hidden" in n,
        "10-step_dit": lambda n: "dit_hidden" in n and "init" not in n,
        "11-output": lambda n: n.startswith("pred_feat") or n.startswith("feat_pred") or n.startswith("vae_decode") or n.startswith("generated_feat"),
    }

    unmatched = []
    for fpath in npy_files:
        name = fpath.name
        arr = np.load(str(fpath))
        matched = False
        for stage_name, matcher in stages.items():
            if matcher(name):
                # Print key statistics
                stats = {
                    "name": name,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "norm": float(np.sqrt(np.sum(arr ** 2))),
                }

                # For step tensors, extract step number
                if name.startswith("step"):
                    step_str = name[4:8]  # e.g., "0000"
                    stats["step"] = int(step_str)

                # First/last few values for comparison
                flat = arr.flatten()
                stats["first_8"] = [float(f"{v:.4f}") for v in flat[:8]]
                if len(flat) > 8:
                    stats["last_8"] = [float(f"{v:.4f}") for v in flat[-8:]]

                log.info("  [%-15s] %s", stage_name, json.dumps(stats, indent=2))
                matched = True
                break
        if not matched:
            unmatched.append(name)

    if unmatched:
        log.info("\nUnmatched files (%d):", len(unmatched))
        for name in unmatched:
            log.info("  - %s", name)

    # Check for audio file
    audio_path = ref_path / "ref_audio.wav"
    if audio_path.exists():
        import soundfile as sf
        audio, sr = sf.read(str(audio_path))
        log.info("\n=== Reference Audio ===")
        log.info("Path: %s", audio_path)
        log.info("Samples: %d", len(audio))
        log.info("Sample rate: %d Hz", sr)
        log.info("Duration: %.2f s", len(audio) / sr)
        log.info("Range: [%.6f, %.6f]", float(audio.min()), float(audio.max()))
        log.info("RMS: %.6f", float(np.sqrt(np.mean(audio ** 2))))


def compute_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened arrays."""
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    dot = np.dot(a_f, b_f)
    norm_a = np.linalg.norm(a_f)
    norm_b = np.linalg.norm(b_f)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean squared error."""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def compute_max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
    """Compute max absolute error."""
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def compare_c_against_ref(ref_dir: str, c_dir: str):
    """Compare C-dumped .npy files against Python reference fixtures."""
    ref_path = Path(ref_dir)
    c_path = Path(c_dir)

    if not ref_path.is_dir():
        log.error("Reference directory not found: %s", ref_dir)
        return
    if not c_path.is_dir():
        log.error("C dump directory not found: %s", c_dir)
        return

    log.info("=== C vs Python Comparison ===")
    log.info("Reference (Python): %s", ref_dir)
    log.info("C dump: %s", c_dir)

    # Match files by name convention
    # C files may have different naming — we try to match by tensor name
    ref_files = {f.name: f for f in ref_path.glob("*.npy")}
    c_files = {f.name: f for f in c_path.glob("*.npy")}

    comparisons = []

    # Try exact name match first
    for name in sorted(set(ref_files.keys()) & set(c_files.keys())):
        ref_arr = np.load(str(ref_files[name]))
        c_arr = np.load(str(c_files[name]))

        if ref_arr.shape != c_arr.shape:
            log.warning("  %s: shape mismatch Python%s vs C%s",
                        name, list(ref_arr.shape), list(c_arr.shape))
            continue

        cos = compute_cosine(ref_arr, c_arr)
        mse = compute_mse(ref_arr, c_arr)
        max_err = compute_max_abs_err(ref_arr, c_arr)

        comparisons.append((name, cos, mse, max_err))

        log.info("  %s: cosine=%.6f MSE=%.6g max_err=%.6g",
                 name, cos, mse, max_err)

    if not comparisons:
        log.warning("No matching files found between %s and %s", ref_dir, c_dir)
        log.info("Reference files: %s", sorted(ref_files.keys())[:10])
        log.info("C files: %s", sorted(c_files.keys())[:10])
        return

    # Summary
    log.info("\n=== Comparison Summary ===")
    cosines = [c[1] for c in comparisons]
    mses = [c[2] for c in comparisons]
    log.info("Matched tensors: %d", len(comparisons))
    log.info("Cosine: min=%.6f max=%.6f mean=%.6f",
             min(cosines), max(cosines), np.mean(cosines))
    log.info("MSE: min=%.6g max=%.6g mean=%.6g",
             min(mses), max(mses), np.mean(mses))

    # Print sorted by cosine (worst first)
    log.info("\n=== Sorted by Cosine (worst first) ===")
    for name, cos, mse, max_err in sorted(comparisons, key=lambda x: x[1]):
        status = "⚠️" if cos < 0.95 else "✅"
        log.info("  %s %s: cos=%.6f MSE=%.6g max_err=%.6g",
                 status, name, cos, mse, max_err)


def main():
    parser = argparse.ArgumentParser(description="C/Python Fixture Comparison")
    parser.add_argument("ref_dir", help="Python reference fixture directory")
    parser.add_argument("--c-dir", default=None,
                        help="C dump directory with .npy files")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze reference only")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.c_dir:
        compare_c_against_ref(args.ref_dir, args.c_dir)
    elif args.analyze:
        analyze_reference(args.ref_dir)
    else:
        analyze_reference(args.ref_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
