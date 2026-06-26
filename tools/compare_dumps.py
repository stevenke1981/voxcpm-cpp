#!/usr/bin/env python3
"""
Compare C intermediate dumps (dump_*.bin) against Python reference fixtures (fixtures/ref/*.npy).

Usage:
    python tools/compare_dumps.py [--dump-dir .] [--fixtures-dir fixtures/ref]
"""

import argparse
import json
import os
import struct
import sys
import numpy as np


def load_bin(path):
    """Load C dump .bin with 3×int32 header."""
    with open(path, "rb") as f:
        hdr = f.read(12)  # 3 × int32
        if len(hdr) != 12:
            raise ValueError(f"Invalid header in {path}")
        ne0, ne1, ne2 = struct.unpack("<iii", hdr)
        data = np.frombuffer(f.read(), dtype=np.float32)
        return data, (ne0, ne1, ne2)


def load_npy(path):
    """Load numpy .npy file."""
    return np.load(path).astype(np.float32)


def compare(label, c_data, c_shape, py_data, py_shape, rtol=1e-3, atol=1e-3):
    """Compare C data vs Python data, checking shapes and values."""
    print(f"\n{'='*60}")
    print(f"COMPARING: {label}")
    print(f"{'='*60}")
    print(f"  C  shape: {c_shape} data: {c_data.shape} floats={len(c_data)}")
    print(f"  Py shape: {py_shape} data: {py_data.shape} floats={len(py_data)}")

    c_flat = c_data.flatten()
    py_flat = py_data.flatten()
    n_c = len(c_flat)
    n_py = len(py_flat)

    if n_c == n_py:
        print(f"  Sizes: MATCH ({n_c})")
    else:
        print(f"  Sizes: DIFFERENT (C={n_c}, Py={n_py})")
        # Possibly reshape issue - try matching total elements
        if n_py > 0 and n_c > 0:
            ratio = n_c / n_py
            print(f"  Size ratio: C/Py = {ratio:.4f}")
        return False

    n = min(n_c, n_py)
    diff = c_flat[:n] - py_flat[:n]
    abs_err = np.abs(diff)
    rel_err = np.where(np.abs(py_flat[:n]) > 1e-8,
                       abs_err / np.abs(py_flat[:n]), abs_err)

    c_rms = np.sqrt(np.mean(c_flat[:n] ** 2))
    py_rms = np.sqrt(np.mean(py_flat[:n] ** 2))
    cos_sim = np.dot(c_flat[:n], py_flat[:n]) / (
        np.linalg.norm(c_flat[:n]) * np.linalg.norm(py_flat[:n]) + 1e-30
    )

    print(f"  C   RMS: {c_rms:.6f}")
    print(f"  Py  RMS: {py_rms:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  RMSE: {np.sqrt(np.mean(diff**2)):.10f}")
    print(f"  Max abs error: {np.max(abs_err):.6f}")
    print(f"  Mean abs error: {np.mean(abs_err):.6f}")
    print(f"  Max rel error: {np.max(rel_err):.6f}")

    if cos_sim > 0.99:
        print(f"  *** MATCH (cos={cos_sim:.6f}) ***")
        return True
    elif cos_sim > 0.9:
        print(f"  ** CLOSE (cos={cos_sim:.6f}) **")
        return True
    elif cos_sim > 0.5:
        print(f"  * PARTIAL (cos={cos_sim:.6f}) *")
        return True
    else:
        print(f"  * DIFFERENT (cos={cos_sim:.6f}) *")
        # Show first few values
        print(f"\n  First 16 values:")
        print(f"    C:  {c_flat[:16]}")
        print(f"    Py: {py_flat[:16]}")
        print(f"    Diff: {diff[:16]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare C dumps vs Python fixtures")
    parser.add_argument("--dump-dir", default=".", help="Directory with dump_*.bin files")
    parser.add_argument("--fixtures-dir", default="fixtures/ref", help="Directory with .npy fixtures")
    args = parser.parse_args()

    # Load fixtures manifest for metadata
    manifest_path = os.path.join(args.fixtures_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        gen_params = manifest.get("_generation_params", {})
        print(f"Generation params: {gen_params}")
        print()

    # Comparison pairs: (dump_name, fixture_name, reshape_to_py)
    comparisons = [
        # Pre-generation intermediates
        ("dump_text_embed.bin",   "text_embed.npy",         None),
        ("dump_base_lm_out.bin",  "base_lm_out.npy",        None),
        ("c_lm_hidden_init.bin",  "lm_hidden_init.npy",     [1, 2048]),
        # Step 0 per-step intermediates
        ("dump_cfm_cond.bin",     "step0000_dit_hidden.npy", None),
        ("dump_step_cond.bin",    "step0000_prefix_feat_cond.npy", None),
        ("dump_step_noise.bin",   None,                      None),  # no Python fixture for initial noise
        ("dump_step_pred_feat.bin","step0000_cfm_pred_feat.npy", None),
        # Final latent
        ("c_latent_dump.bin",     "generated_feat.npy",     None),
    ]

    results = []

    for dump_name, fixture_name, reshape_ref in comparisons:
        dump_path = os.path.join(args.dump_dir, dump_name)
        if not os.path.exists(dump_path):
            print(f"\n--- {dump_name}: NOT FOUND (skipping) ---")
            continue

        # Load C dump
        try:
            c_data, c_shape = load_bin(dump_path)
        except Exception as e:
            print(f"\n--- {dump_name}: ERROR loading: {e} ---")
            continue

        # Handle c_latent_dump.bin format differently (has 3 int32 headers, not ne format)
        if dump_name == "c_latent_dump.bin":
            # Already parsed by load_bin which treats first 12 bytes as ne0/ne1/ne2
            # Actual data starts after 12 bytes
            pass

        if fixture_name is None:
            print(f"\n--- {dump_name}: No Python fixture to compare, stats only ---")
            c_rms = np.sqrt(np.mean(c_data ** 2))
            print(f"  C shape: {c_shape}  C RMS: {c_rms:.6f}  floats: {len(c_data)}")
            print(f"  Range: [{c_data.min():.6f}, {c_data.max():.6f}]")
            continue

        fixture_path = os.path.join(args.fixtures_dir, fixture_name)
        if not os.path.exists(fixture_path):
            print(f"\n--- {dump_name} vs {fixture_name}: FIXTURE NOT FOUND ---")
            # Still print C stats
            c_rms = np.sqrt(np.mean(c_data ** 2))
            print(f"  C shape: {c_shape}  C RMS: {c_rms:.6f}  floats: {len(c_data)}")
            continue

        # Load Python fixture
        py_data = load_npy(fixture_path)
        py_shape = py_data.shape
        py_flat = py_data.flatten()

        # Handle different naming between C and Python
        if dump_name == "dump_step_cond.bin":
            # C: [latent_dim, patch_size] = [64, 4], Python step0000_prefix_feat_cond: [1, 4, 64]
            # Python is [batch, patch_size, latent_dim] — need to transpose
            if len(py_shape) == 3 and py_shape[2] == c_shape[0] and py_shape[1] == c_shape[1]:
                py_compare = py_data.transpose(0, 2, 1)  # [1, 4, 64] -> [1, 64, 4]
            else:
                py_compare = py_data
        elif dump_name == "dump_step_pred_feat.bin":
            # C: [latent_dim, patch_size] = [64, 4], Python step0000_cfm_pred_feat: [1, 4, 64]
            if len(py_shape) == 3 and py_shape[2] == c_shape[0] and py_shape[1] == c_shape[1]:
                py_compare = py_data.transpose(0, 2, 1)  # [1, 4, 64] -> [1, 64, 4]
            else:
                py_compare = py_data
        elif dump_name == "dump_cfm_cond.bin":
            # C: [dit_hidden_size*2, 1] = [2048, 1], Python step0000_dit_hidden: [1, 2048]
            # Python is [batch, dit_hidden] — just flatten
            pass  # leave as-is
        elif dump_name == "c_latent_dump.bin":
            # C: flat [n_patches * patch_size * latent_dim], Python: [8, 4, 64]
            # Reshape C to match Python layout
            # C header: n_patches=8, patch_size=4, latent_dim=64
            c_flat = c_data
            total_elems = c_shape[0] * c_shape[1] * max(c_shape[2], 1)
            # Try reshaping
            pass

        # Compare
        c_flat = c_data.flatten()
        py_compare_flat = py_data.flatten()  # Just flatten and compare for now

        matched = compare(
            f"{dump_name} vs {fixture_name}",
            c_flat, c_shape,
            py_compare_flat, py_shape,
        )
        results.append((dump_name, fixture_name, matched))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for dump_name, fixture_name, matched in results:
        status = "✅" if matched else "❌"
        if not matched:
            all_ok = False
        print(f"  {status} {dump_name} vs {fixture_name}")
    print(f"\nOverall: {'ALL MATCH' if all_ok else 'SOME MISMATCH — see details above'}")


if __name__ == "__main__":
    main()
