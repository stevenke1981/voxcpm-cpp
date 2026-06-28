#!/usr/bin/env python3
"""
Objective audio quality analysis for VoxCPM-C generated WAV files.

Usage:
    python tools/analyze_audio_quality.py <wav_path_1> [wav_path_2 ...]
"""

import argparse
import json
import os
import sys
import numpy as np
from scipy import signal, ndimage
import soundfile as sf


def analyze_wav(path):
    """Analyze a single WAV file and return quality metrics."""
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data[:, 0]  # mono
    duration = len(data) / sr

    # --- Time-domain metrics ---
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data ** 2)))
    crest = peak / max(rms, 1e-12)  # peak-to-RMS ratio (crest factor)
    zero_cross = float(np.mean(np.abs(np.diff(np.sign(data)))) / 2)  # ZCR (fraction)
    n_nan = int(np.sum(~np.isfinite(data)))
    n_clip = int(np.sum(np.abs(data) >= 0.999))
    n_silent = int(np.sum(np.abs(data) < 1e-6))
    silent_frac = n_silent / len(data)

    # Dynamic range: ratio of 95th to 5th percentile amplitude
    amps = np.abs(data)
    p5 = float(np.percentile(amps, 5) + 1e-12)
    p95 = float(np.percentile(amps, 95))
    dynamic_range_db = 20 * np.log10(p95 / max(p5, 1e-12))

    # --- Frequency-domain metrics ---
    nperseg = min(1024, len(data) // 4)
    f, t, Sxx = signal.spectrogram(data, sr, nperseg=nperseg, noverlap=nperseg // 2)
    # Avoid log(0)
    Sxx = np.maximum(Sxx, 1e-12)

    # Spectral centroid (weighted mean frequency)
    freqs = f.reshape(-1, 1)
    spec_centroid = float(np.sum(freqs * Sxx) / np.sum(Sxx))

    # Spectral bandwidth (spread around centroid)
    diff = (freqs - spec_centroid) ** 2
    spec_bandwidth = float(np.sqrt(np.sum(diff * Sxx) / np.sum(Sxx)))

    # Spectral flatness (geometric/arithmetic mean ratio - low = tonal, high = noise-like)
    # Compute per frame then average
    flatness_per_frame = []
    for i in range(Sxx.shape[1]):
        frame = Sxx[:, i]
        gmean = np.exp(np.mean(np.log(np.maximum(frame, 1e-12))))
        amean = np.mean(frame)
        flatness_per_frame.append(gmean / max(amean, 1e-12))
    spectral_flatness = float(np.mean(flatness_per_frame))

    # Spectral roll-off: frequency below which 85% of energy is contained
    cum_energy = np.cumsum(Sxx, axis=0)
    total_energy = cum_energy[-1, :]
    rolloff_85 = []
    for i in range(Sxx.shape[1]):
        idx = np.searchsorted(cum_energy[:, i], 0.85 * max(total_energy[i], 1e-12))
        rolloff_85.append(float(f[min(idx, len(f) - 1)]))
    rolloff_85_hz = float(np.mean(rolloff_85))

    # --- Noise floor estimation (from silent portions) ---
    # Look for the quietest 10% of frames
    frame_energies = np.sum(Sxx, axis=0)
    quiet_thresh = np.percentile(frame_energies, 10)
    quiet_frames = Sxx[:, frame_energies <= quiet_thresh]
    noise_floor = float(np.mean(quiet_frames)) if quiet_frames.size > 0 else 0.0
    noise_floor_db = 10 * np.log10(max(noise_floor, 1e-20))

    # Signal-to-noise ratio estimate (peak signal / noise floor)
    peak_psd = float(np.max(Sxx))
    snr_estimate_db = 10 * np.log10(max(peak_psd / max(noise_floor, 1e-20), 1e-20))

    # --- Speech detection heuristics ---
    # Speech typically has: moderate ZCR, non-flat spectrum, temporal modulation
    is_speech_like = (
        0.02 < zero_cross < 0.4 and
        spectral_flatness < 0.5 and
        dynamic_range_db > 12 and
        crest > 3
    )

    metrics = {
        "file": os.path.basename(path),
        "duration_s": round(duration, 3),
        "sample_rate": sr,
        "samples": len(data),
        "time_domain": {
            "peak": round(peak, 6),
            "rms": round(rms, 6),
            "crest_factor": round(crest, 4),
            "dynamic_range_db": round(dynamic_range_db, 2),
            "zero_crossing_rate": round(zero_cross, 6),
            "clip_count": n_clip,
            "nan_count": n_nan,
            "silent_fraction": round(silent_frac, 6),
        },
        "spectral": {
            "centroid_hz": round(spec_centroid, 1),
            "bandwidth_hz": round(spec_bandwidth, 1),
            "rolloff_85_hz": round(rolloff_85_hz, 1),
            "flatness": round(spectral_flatness, 6),
            "noise_floor_db": round(noise_floor_db, 2),
            "snr_estimate_db": round(snr_estimate_db, 2),
        },
        "classification": {
            "speech_like": is_speech_like,
            "noisy": spectral_flatness > 0.5,
            "clipped": n_clip > 0,
            "has_nan": n_nan > 0,
            "too_quiet": rms < 0.001,
            "silent": silent_frac > 0.9,
        },
    }
    return metrics


def compare_wavs(paths):
    """Compare multiple WAV files for consistency."""
    if len(paths) < 2:
        return None

    print("\n" + "=" * 60)
    print("CROSS-FILE COMPARISON")
    print("=" * 60)

    datas = []
    for p in paths:
        d, sr = sf.read(p)
        if d.ndim > 1:
            d = d[:, 0]
        datas.append((os.path.basename(p), d, sr))

    # Compare pairwise
    for i in range(len(datas)):
        for j in range(i + 1, len(datas)):
            n1, d1, sr1 = datas[i]
            n2, d2, sr2 = datas[j]
            print(f"\n  {n1} vs {n2}:")
            if sr1 != sr2:
                print(f"    Sample rate mismatch: {sr1} vs {sr2}")
                continue

            min_len = min(len(d1), len(d2))
            if min_len == 0:
                print("    Both empty")
                continue

            d1_t = d1[:min_len]
            d2_t = d2[:min_len]

            diff = d1_t - d2_t
            cos = np.dot(d1_t, d2_t) / (np.linalg.norm(d1_t) * np.linalg.norm(d2_t) + 1e-30)
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            max_err = float(np.max(np.abs(diff)))
            snr_db = 20 * np.log10(np.linalg.norm(d1_t) / max(np.linalg.norm(diff), 1e-30))

            print(f"    Cosine similarity: {cos:.6f}")
            print(f"    RMSE: {rmse:.8f}")
            print(f"    Max error: {max_err:.6f}")
            print(f"    SNR (dB): {snr_db:.2f}")
            print(f"    Length match: {'YES' if len(d1) == len(d2) else f'C={len(d1)} Py={len(d2)}'}")

            if cos > 0.99:
                print(f"    => ESSENTIALLY IDENTICAL")
            elif cos > 0.9:
                print(f"    => VERY SIMILAR")
            elif cos > 0.5:
                print(f"    => PARTIALLY CORRELATED")
            else:
                print(f"    => UNCORRELATED")


def main():
    parser = argparse.ArgumentParser(description="Analyze VoxCPM-C audio quality")
    parser.add_argument("wavs", nargs="+", help="WAV file(s) to analyze")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    all_metrics = []

    for wav_path in args.wavs:
        if not os.path.exists(wav_path):
            print(f"ERROR: {wav_path} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"ANALYZING: {wav_path}")
        print(f"{'=' * 60}")

        metrics = analyze_wav(wav_path)
        all_metrics.append(metrics)

        if args.json:
            continue

        # Print readable summary
        td = metrics["time_domain"]
        sp = metrics["spectral"]
        cl = metrics["classification"]

        print(f"  Duration: {metrics['duration_s']:.2f}s @ {metrics['sample_rate']} Hz")
        print(f"  Samples:  {metrics['samples']}")
        print(f"")
        print(f"  Time Domain:")
        print(f"    Peak:       {td['peak']:.6f}")
        print(f"    RMS:        {td['rms']:.6f}")
        print(f"    Crest:      {td['crest_factor']:.2f}x")
        print(f"    Dynamic R:  {td['dynamic_range_db']:.1f} dB")
        print(f"    ZCR:        {td['zero_crossing_rate']:.4f}")
        print(f"    Clip:       {td['clip_count']}")
        print(f"    NaN:        {td['nan_count']}")
        print(f"    Silence:    {td['silent_fraction']*100:.2f}%")
        print(f"")
        print(f"  Spectral:")
        print(f"    Centroid:   {sp['centroid_hz']:.0f} Hz")
        print(f"    Bandwidth:  {sp['bandwidth_hz']:.0f} Hz")
        print(f"    Rolloff85:  {sp['rolloff_85_hz']:.0f} Hz")
        print(f"    Flatness:   {sp['flatness']:.6f}")
        print(f"    Noise flr:  {sp['noise_floor_db']:.1f} dB")
        print(f"    Est SNR:    {sp['snr_estimate_db']:.1f} dB")
        print(f"")
        print(f"  Classification:")
        for k, v in cl.items():
            print(f"    {k}: {'⚠️' if v else '✓'}" if v and k in ('noisy', 'clipped', 'has_nan', 'too_quiet', 'silent')
                  else f"    {k}: {'YES' if v else 'no'}")

    # Compare if multiple files
    compare_wavs(args.wavs)

    if args.json:
        print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
