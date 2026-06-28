#!/usr/bin/env python3
"""Generate spectrogram PNG for VoxCPM-C WAV files."""
import sys, os
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def gen_spectrogram(wav_path, out_path=None):
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]
    if out_path is None:
        out_path = os.path.splitext(wav_path)[0] + '_spec.png'

    nperseg = min(1024, len(data) // 4)
    f, t, Sxx = signal.spectrogram(data, sr, nperseg=nperseg, noverlap=nperseg // 2)
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-12))

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_ylim(0, min(sr//2, 8000))
    fig.colorbar(im, ax=ax, label='dB')
    ax.set_title(f'{os.path.basename(wav_path)} — {len(data)/sr:.2f}s @ {sr}Hz')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}  ({Sxx.shape[1]} frames, {Sxx.shape[0]} bins)")

if __name__ == '__main__':
    for path in sys.argv[1:]:
        if os.path.exists(path):
            gen_spectrogram(path)
        else:
            print(f"Not found: {path}")
