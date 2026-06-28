#!/usr/bin/env python3
"""Detailed WAV analysis for diagnosing unintelligible speech."""
import sys, os, json
import numpy as np
import soundfile as sf

def analyze(path):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data[:, 0]

    dur = len(data) / sr
    print(f"\n{'='*60}")
    print(f"FILE: {os.path.basename(path)}")
    print(f"{'='*60}")
    print(f"Duration: {dur:.3f}s @ {sr}Hz  Samples: {len(data)}")

    # Time-domain structure
    print(f"\n--- Time Domain ---")
    print(f"  Peak: {np.max(np.abs(data)):.6f}")
    print(f"  RMS:  {np.sqrt(np.mean(data**2)):.6f}")
    print(f"  Mean: {np.mean(data):.6f}")

    # Check for repeating patterns (periodicity)
    # Auto-correlation for lags 1-1000
    if len(data) > 2000:
        segment = data[:min(len(data), 48000)]  # first 1s
        seg = segment - np.mean(segment)
        ac = np.correlate(seg, seg, mode='full')
        ac = ac[len(ac)//2:]  # positive lags only
        ac = ac / ac[0]  # normalize
        peaks = []
        for i in range(100, min(1000, len(ac))):
            if ac[i] > ac[i-1] and ac[i] > ac[i+1] and ac[i] > 0.1:
                peaks.append((i, ac[i]))
        if peaks:
            top = sorted(peaks, key=lambda x: -x[1])[:5]
            print(f"\n  Top autocorrelation peaks (lag, correlation):")
            for lag, corr in top:
                freq = sr / lag
                print(f"    lag={lag:4d} samples  corr={corr:.3f}  freq={freq:.1f} Hz")

    # Frame-by-frame energy analysis
    frame_len = 256
    n_frames = len(data) // frame_len
    frame_energies = []
    for i in range(n_frames):
        f = data[i*frame_len:(i+1)*frame_len]
        frame_energies.append(np.sqrt(np.mean(f**2)))
    frame_energies = np.array(frame_energies)

    print(f"\n  Frame energy ({frame_len} samples/frame):")
    print(f"    Mean: {np.mean(frame_energies):.6f}")
    print(f"    Std:  {np.std(frame_energies):.6f}")
    print(f"    Max:  {np.max(frame_energies):.6f}")
    print(f"    Min:  {np.min(frame_energies):.6f}")

    # Check for alternating sign pattern (high-frequency oscillation)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(data)))) / 2
    zcr = zero_crossings / len(data)
    print(f"\n  Zero-crossing rate: {zcr:.4f}")

    # Check if it's mostly silence with short bursts
    loud_mask = np.abs(data) > np.max(np.abs(data)) * 0.1
    loud_frac = np.sum(loud_mask) / len(data)
    print(f"  Fraction > 10% peak: {loud_frac*100:.1f}%")

    # Print first few and last few samples
    print(f"\n  First 20 samples: {data[:20].tolist()}")
    print(f"  Samples 1000-1020: {data[1000:1020].tolist()}")
    if len(data) > 10000:
        print(f"  Samples 10000-10020: {data[10000:10020].tolist()}")

    # Spectral analysis via simple FFT
    print(f"\n--- Spectral (FFT of whole signal) ---")
    fft = np.fft.rfft(data)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(len(data), 1/sr)

    # Dominant frequencies
    top_n = 10
    top_indices = np.argsort(mag)[-top_n:][::-1]
    print(f"  Top {top_n} frequencies:")
    for idx in top_indices[:10]:
        if freqs[idx] > 0:
            print(f"    {freqs[idx]:7.1f} Hz  mag={mag[idx]:.1f}")

    # Energy distribution
    bands = [(0, 300, "Sub-bass"), (300, 800, "Bass/mid"),
             (800, 2000, "Lower formant"), (2000, 4000, "Upper formant"),
             (4000, 8000, "Sibilance"), (8000, sr//2, "High freq")]
    total_energy = np.sum(mag**2)
    print(f"\n  Energy per band:")
    for lo, hi, name in bands:
        mask = (freqs >= lo) & (freqs < hi)
        e = np.sum(mag[mask]**2) / max(total_energy, 1e-30) * 100
        print(f"    {name:20s} ({lo:5d}-{hi:5d} Hz): {e:.1f}%")

    # Check waveform shape - is it symmetric?
    pos_frac = np.sum(data > 0) / len(data)
    print(f"\n  Positive fraction: {pos_frac*100:.1f}%")

    # Check for clipping
    n_clip = np.sum(np.abs(data) >= 0.999)
    print(f"  Clipped samples: {n_clip}")

    # Check for sudden jumps (artifacts)
    diffs = np.abs(np.diff(data))
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    print(f"  Max adjacent diff: {max_diff:.6f}")
    print(f"  Mean adjacent diff: {mean_diff:.6f}")
    big_jumps = np.sum(diffs > mean_diff * 10)
    print(f"  Jumps > 10x mean: {big_jumps} / {len(diffs)}")

    return data, sr

if __name__ == '__main__':
    for path in sys.argv[1:]:
        if os.path.exists(path):
            analyze(path)
