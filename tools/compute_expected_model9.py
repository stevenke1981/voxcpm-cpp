"""Compute expected model.9 output using Python reference data."""
import numpy as np
import os

file_size = os.path.getsize('E:/voxcpm-cpp/fixtures/ref/vae_decode_raw.npy')
print(f'vae_decode_raw.npy size: {file_size} bytes')
vae_decode_raw = np.load('E:/voxcpm-cpp/fixtures/ref/vae_decode_raw.npy')
print(f'vae_decode_raw shape: {vae_decode_raw.shape}')
print(f'vae_decode_raw RMS: {np.sqrt(np.mean(vae_decode_raw**2)):.6f}')

latent = np.load('E:/voxcpm-cpp/fixtures/ref/feat_pred_latent.npy')
print(f'\nfeat_pred_latent shape: {latent.shape}')
print(f'feat_pred_latent RMS: {np.sqrt(np.mean(latent**2)):.6f}')
