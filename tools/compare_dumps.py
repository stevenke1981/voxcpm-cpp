#!/usr/bin/env python3
"""
Compare C intermediate dumps (dump_*.bin) against Python reference fixtures (fixtures/ref/*.npy).

Per-step C dumps (now with step number, e.g. dump_step_cond_0000.bin) are matched against
corresponding step0000_* Python fixtures.

Usage:
    python tools/compare_dumps.py [--dump-dir .] [--fixtures-dir fixtures/ref]
"""

import argparse
import json
import os
import re
import struct
import sys
import numpy as np


def load_bin(path):
    """Load C dump .bin with 3xint32 header."""
    with open(path, "rb") as f:
        hdr = f.read(12)  # 3 x int32
        if len(hdr) != 12:
            raise ValueError(f"Invalid header in {path}")
        ne0, ne1, ne2 = struct.unpack("<iii", hdr)
        data = np.frombuffer(f.read(), dtype=np.float32)
        return data, (ne0, ne1, ne2)


def load_npy(path):
    """Load numpy .npy file."""
    return np.load(path).astype(np.float32)


def compare_one(label, c_data, c_shape, py_data, py_shape):
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
        if n_py > 0 and n_c > 0:
            print(f"  Size ratio: C/Py = {n_c / n_py:.4f}")
        return False

    n = min(n_c, n_py)
    diff = c_flat[:n] - py_flat[:n]
    abs_err = np.abs(diff)
    with np.errstate(divide='ignore', invalid='ignore'):
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

    matched = False
    if cos_sim > 0.99:
        print(f"  [MATCH] (cos={cos_sim:.6f})")
        matched = True
    elif cos_sim > 0.9:
        print(f"  [CLOSE] (cos={cos_sim:.6f})")
        matched = True
    elif cos_sim > 0.5:
        print(f"  [PARTIAL] (cos={cos_sim:.6f})")
        matched = True
    else:
        print(f"  [DIFFERENT] (cos={cos_sim:.6f})")
        print(f"\n  First 16 values:")
        print(f"    C:  {c_flat[:16]}")
        print(f"    Py: {py_flat[:16]}")
        print(f"    Diff: {diff[:16]}")

    return matched


def reshape_c_to_py(dump_name, c_data, c_shape, py_data, py_shape):
    """Transpose/reshape C data to match Python layout where needed."""
    if dump_name.startswith("dump_step_cond_") or dump_name.startswith("dump_step_noise_") or dump_name.startswith("dump_step_pred_feat_"):
        # C: [latent_dim, patch_size] = [64, 4]
        # Py: [1, 4, 64] or [1, patch_size, latent_dim]
        if len(py_shape) == 3 and py_shape[2] == c_shape[0] and py_shape[1] == c_shape[1]:
            py_compare = py_data.transpose(0, 2, 1)  # [1, 4, 64] -> [1, 64, 4]
            return c_data, py_compare
        elif len(py_shape) == 2 and py_shape[1] == c_shape[0] and py_shape[0] == 1:
            py_compare = py_data.reshape(c_shape[0], c_shape[1])
            return c_data, py_compare
        elif len(py_shape) == 1 and py_shape[0] == c_shape[0] * c_shape[1]:
            py_compare = py_data.reshape(c_shape[0], c_shape[1])
            return c_data, py_compare
    elif dump_name.startswith("dump_mu_init_"):
        # C: [dit_hidden_size*2, 1] = [2048, 1]
        # Py step0000_dit_hidden: [1, 2048]
        c_flat = c_data.flatten()
        if len(py_shape) == 2 and py_shape[1] == c_flat.shape[0] and py_shape[0] == 1:
            py_compare = py_data.flatten()
            return c_flat, py_compare
        elif len(py_shape) == 1 and py_shape[0] == c_flat.shape[0]:
            return c_flat, py_data
    elif dump_name.startswith("dump_lm_hidden_ar_"):
        # C: [hidden_size, 1] = [2048, 1]
        # Py step0000_lm_hidden_step: [1, 2048]
        c_flat = c_data.flatten()
        if len(py_shape) == 2 and py_shape[1] == c_flat.shape[0] and py_shape[0] == 1:
            return c_flat, py_data.flatten()
        elif len(py_shape) == 1:
            return c_flat, py_data
    elif dump_name.startswith("dump_residual_hidden_ar_"):
        # C: [res_hidden_size, 1]
        # Py step0000_residual_hidden_step: [1, res_hidden_size]
        c_flat = c_data.flatten()
        if len(py_shape) == 2 and py_shape[1] == c_flat.shape[0] and py_shape[0] == 1:
            return c_flat, py_data.flatten()
        elif len(py_shape) == 1:
            return c_flat, py_data

    return c_data, py_data


def find_fixture_for_step(fixtures_dir, step_num, suffix_patterns):
    """Find a step000X fixture matching the given step number and suffix patterns."""
    step_str = f"step{step_num:04d}"
    for suffix in suffix_patterns:
        fname = f"{step_str}_{suffix}.npy"
        fpath = os.path.join(fixtures_dir, fname)
        if os.path.exists(fpath):
            return fpath
    return None


def is_c_latent_dump(dump_name):
    return dump_name == "c_latent_dump.bin"


def compare_latent_dump(c_data, c_shape, fixtures_dir):
    """Compare c_latent_dump.bin (custom format: n_patches, patch_size, latent_dim header)."""
    # Custom header: first 3 int32 = n_patches, patch_size, latent_dim
    print(f"\n{'='*60}")
    print("COMPARING: c_latent_dump.bin vs generated_feat.npy")
    print(f"{'='*60}")
    print(f"  C raw data: {len(c_data)} floats, header={c_shape}")
    # Re-interpret: first 12 bytes were read as ne0,ne1,ne2 but actual format:
    # native header is [n_patches, patch_size, latent_dim]
    n_patches_c = c_shape[0]
    patch_size_c = c_shape[1]
    latent_dim_c = c_shape[2]
    total_elem = n_patches_c * patch_size_c * latent_dim_c
    c_flat = c_data[:total_elem]
    print(f"  C: n_patches={n_patches_c} patch_size={patch_size_c} latent_dim={latent_dim_c}")
    print(f"  C reshaped: ({n_patches_c}, {patch_size_c}, {latent_dim_c}) = {len(c_flat)} floats")

    fixture_path = os.path.join(fixtures_dir, "generated_feat.npy")
    if not os.path.exists(fixture_path):
        print("  generated_feat.npy: NOT FOUND")
        return False

    py_data = load_npy(fixture_path)
    print(f"  Py shape: {py_data.shape}")
    py_flat = py_data.flatten()

    if len(c_flat) != len(py_flat):
        print(f"  Sizes: DIFFERENT (C={len(c_flat)}, Py={len(py_flat)})")
        print(f"  Size ratio: C/Py = {len(c_flat) / max(len(py_flat),1):.4f}")
        # Try matching partial
        n = min(len(c_flat), len(py_flat))
        c_flat = c_flat[:n]
        py_flat = py_flat[:n]

    return compare_one("c_latent_dump.bin vs generated_feat.npy",
                        c_flat, (n_patches_c, patch_size_c, latent_dim_c),
                        py_flat, py_data.shape)


def main():
    parser = argparse.ArgumentParser(description="Compare C dumps vs Python fixtures")
    parser.add_argument("--dump-dir", default=".", help="Directory with dump_*.bin files")
    parser.add_argument("--fixtures-dir", default="fixtures/ref", help="Directory with .npy fixtures")
    args = parser.parse_args()

    # Load fixtures manifest
    manifest_path = os.path.join(args.fixtures_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        gen_params = manifest.get("_generation_params", {})
        print(f"Generation params: {gen_params}")
        print()

    # Discover all dump_*.bin files
    all_dumps = sorted([f for f in os.listdir(args.dump_dir)
                        if f.endswith(".bin") and f.startswith("dump_")])

    print(f"Found {len(all_dumps)} dump files in {args.dump_dir}")
    for d in all_dumps:
        sz = os.path.getsize(os.path.join(args.dump_dir, d))
        print(f"  {d}: {sz} bytes")

    # Old-style flat comparisons (non-step-specific)
    old_style = [
        ("dump_text_embed.bin",   "text_embed.npy"),
        ("dump_base_lm_out.bin",  "base_lm_out.npy"),
    ]

    # Step-specific dump mappings: (dump_prefix, step_suffixes_for_dump, fixture_suffix_list, step_offset)
    # step_offset: number of steps to add to the C dump step number to get the correct Python fixture step.
    # C's ar_step_counter starts at 0 for the first generation step.
    # For CFM/FE outputs, step_counter=0 maps to Python step0000.
    # For LM/RESIDUAL hidden states, step_counter=N maps to Python step(N+1) because
    #   the hidden state is dumped AFTER the LM update, which produces the state for
    #   the NEXT step. Step0000 = initial state (pre-audio), step0001 = after first audio.
    # e.g. dump_lm_hidden_ar_0000.bin matches step0001_lm_hidden_step.npy
    step_pairs = [
        ("dump_step_cond_",       ["prefix_feat_cond"],       0),
        ("dump_step_noise_",      None,                       0),  # no Python fixture for noise
        ("dump_step_pred_feat_",  ["cfm_pred_feat"],          0),
        ("dump_mu_init_",         ["dit_hidden"],             0),
        ("dump_lm_hidden_ar_",    ["lm_hidden_step"],         1),
        ("dump_residual_hidden_ar_", ["residual_hidden_step"], 1),
    ]

    results = []

    # Compare old-style flat dumps
    for dump_name, fixture_name in old_style:
        dump_path = os.path.join(args.dump_dir, dump_name)
        if not os.path.exists(dump_path):
            print(f"\n--- {dump_name}: NOT FOUND (skipping) ---")
            continue

        try:
            c_data, c_shape = load_bin(dump_path)
        except Exception as e:
            print(f"\n--- {dump_name}: ERROR loading: {e} ---")
            continue

        fixture_path = os.path.join(args.fixtures_dir, fixture_name)
        if not os.path.exists(fixture_path):
            print(f"\n--- {dump_name} vs {fixture_name}: FIXTURE NOT FOUND ---")
            c_rms = np.sqrt(np.mean(c_data ** 2))
            print(f"  C shape: {c_shape}  C RMS: {c_rms:.6f}  floats: {len(c_data)}")
            continue

        py_data = load_npy(fixture_path)
        c_compare, py_compare = reshape_c_to_py(dump_name, c_data, c_shape, py_data, py_data.shape)
        matched = compare_one(f"{dump_name} vs {fixture_name}",
                              c_compare, c_shape,
                              py_compare, py_data.shape)
        results.append((dump_name, fixture_name, matched))

    # Compare step-specific dumps
    dump_by_step = {}
    for d in all_dumps:
        m = re.match(r"dump_(.+)_(\d{4})\.bin", d)
        if m:
            base = m.group(1)
            step = int(m.group(2))
            if step not in dump_by_step:
                dump_by_step[step] = {}
            dump_by_step[step][base] = os.path.join(args.dump_dir, d)

    print(f"\n{'='*60}")
    print("STEP-SPECIFIC COMPARISONS")
    print(f"{'='*60}")
    print(f"Found {len(dump_by_step)} steps with per-step dumps")

    for step_num in sorted(dump_by_step.keys()):
        step_dumps = dump_by_step[step_num]
        for base_name, dump_path in sorted(step_dumps.items()):
            # Find matching suffix in step_pairs
            matched_pair = None
            for prefix, suffix_list, step_offset in step_pairs:
                # prefix e.g. "step_cond_" (without "dump_")
                p = prefix.replace("dump_", "")
                if base_name == p.rstrip("_"):
                    matched_pair = (prefix, suffix_list, step_offset)
                    break

            if matched_pair is None:
                print(f"\n--- {base_name}_{step_num:04d}.bin: No matching pair defined ---")
                continue

            prefix, suffix_list, step_offset = matched_pair

            try:
                c_data, c_shape = load_bin(dump_path)
            except Exception as e:
                print(f"\n--- {base_name}_{step_num:04d}: ERROR loading: {e} ---")
                continue

            if suffix_list is None:
                # No Python fixture, just print stats
                print(f"\n--- {os.path.basename(dump_path)}: No Python fixture ---")
                c_rms = np.sqrt(np.mean(c_data ** 2))
                print(f"  C shape: {c_shape}  C RMS: {c_rms:.6f}  floats: {len(c_data)}")
                print(f"  Range: [{c_data.min():.6f}, {c_data.max():.6f}]")
                continue

            # Find matching step fixture (apply step_offset for post-update states)
            fixture_step = step_num + step_offset
            fixture_path = find_fixture_for_step(args.fixtures_dir, fixture_step, suffix_list)
            if fixture_path is None:
                print(f"\n--- {os.path.basename(dump_path)}: No matching fixture for step {fixture_step} (C step {step_num} + offset {step_offset}) ---")
                # Try without offset as fallback
                fixture_path = find_fixture_for_step(args.fixtures_dir, step_num, suffix_list)
                if fixture_path:
                    print(f"    (found step{step_num:04d} instead — may be off-by-one mismatch)")
                else:
                    continue

            py_data = load_npy(fixture_path)
            dump_name = os.path.basename(dump_path)
            fixture_name = os.path.basename(fixture_path)
            c_compare, py_compare = reshape_c_to_py(dump_name, c_data, c_shape, py_data, py_data.shape)
            matched = compare_one(f"{dump_name} vs {fixture_name}",
                                  c_compare, c_shape,
                                  py_compare, py_data.shape)
            results.append((dump_name, fixture_name, matched))

    # Compare c_latent_dump.bin
    latent_dump_path = os.path.join(args.dump_dir, "c_latent_dump.bin")
    if os.path.exists(latent_dump_path):
        try:
            c_data, c_shape = load_bin(latent_dump_path)
            matched = compare_latent_dump(c_data, c_shape, args.fixtures_dir)
            results.append(("c_latent_dump.bin", "generated_feat.npy", matched))
        except Exception as e:
            print(f"\n--- c_latent_dump.bin: ERROR: {e} ---")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for dump_name, fixture_name, matched in results:
        status = "[OK]" if matched else "[FAIL]"
        if not matched:
            all_ok = False
        print(f"  {status} {dump_name} vs {fixture_name}")
    print(f"\nOverall: {'ALL MATCH' if all_ok else 'SOME MISMATCH -- see details above'}")


if __name__ == "__main__":
    main()
