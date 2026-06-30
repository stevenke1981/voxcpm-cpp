# Pinned Converter and GGUF Contract (2026-06-30)

## Pinned upstream

Converter compatibility is locked to:

```text
openbmb/VoxCPM2
bffb3df5a29440629464e5e839f4d214c8714c3d
```

[`tests/converter/hf_snapshot.lock.json`](../tests/converter/hf_snapshot.lock.json)
records SHA-256 values for config, tokenizer, safetensors, and AudioVAE files.
CI downloads only the small config/tokenizer subset from that exact revision
and checks the hashes. A local full-snapshot conversion can use the same lock,
which also verifies both multi-gigabyte weight files before conversion.

This is intentionally a commit hash, not `main`: upstream changes cannot alter
converter inputs silently.

## Contract gates

The converter now:

- verifies every file named by `--snapshot-lock`;
- records `voxcpm.source_repo` and `voxcpm.source_revision` in GGUF metadata;
- rejects duplicate canonical names;
- rejects generic/fallthrough tensor mappings unless
  `--allow-unmapped` is explicitly supplied;
- writes residual-LM dimensions and Base-LM head dimension instead of relying
  on C loader defaults;
- emits correct reversed GGUF shapes for tensors of every rank;
- rejects misleading direct Q8/Q4 output and directs users to convert to
  F16/F32 followed by the native `quantize` executable.

[`tools/validate_gguf_contract.py`](../tools/validate_gguf_contract.py) validates
required metadata, source revision, tokenizer metadata, tensor uniqueness,
positive shapes, all configured Base LM/RALM/LocEnc/LocDiT layer families,
required top-level tensors, embedding dimensions, emitted shapes, and tensor
types.

## Weight-free CI smoke

[`tests/converter/create_pinned_snapshot.py`](../tests/converter/create_pinned_snapshot.py)
creates deterministic tiny safetensors using the real upstream module names.
The converter writes an actual GGUF from that snapshot, the validator checks
the result and shape manifest, and a deliberately corrupted lock must fail.
This keeps the normal CI test small while still exercising file I/O, name
mapping, metadata, dtype, shape, and integrity paths.

The dedicated GitHub Actions job installs only `numpy`, `safetensors`, and
`gguf`; PyTorch is loaded lazily and is required only when a real
`audiovae.pth` is converted.

## Commands

Fast CI-equivalent checks:

```powershell
python .\tests\converter\verify_pinned_hf_snapshot.py
python .\tests\converter\test_converter_contract.py
```

Full local source-snapshot preflight, without writing a multi-gigabyte GGUF:

```powershell
python .\tools\convert_voxcpm2_to_gguf.py `
  --hf-dir .\pretrained_models\VoxCPM2 `
  --out .\models\voxcpm2-f16.gguf `
  --snapshot-lock .\tests\converter\hf_snapshot.lock.json `
  --contract-only
```

Conversion and output validation:

```powershell
python .\tools\convert_voxcpm2_to_gguf.py `
  --hf-dir .\pretrained_models\VoxCPM2 `
  --out .\models\voxcpm2-f16.gguf `
  --outtype f16 `
  --snapshot-lock .\tests\converter\hf_snapshot.lock.json `
  --emit-shape-manifest .\models\voxcpm2-f16.shapes.json

python .\tools\validate_gguf_contract.py `
  --gguf .\models\voxcpm2-f16.gguf `
  --shape-manifest .\models\voxcpm2-f16.shapes.json
```

The lock is expected to change only when the project deliberately adopts a
new upstream revision and regenerates/reviews the converter contract.
