#!/usr/bin/env python3
"""
Placeholder converter for VoxCPM2 -> GGUF.

This file intentionally contains only the CLI skeleton. The real implementation
must use gguf writer utilities and safetensors readers, then follow
model_conversion.md.
"""
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert VoxCPM2 HF snapshot to GGUF")
    parser.add_argument("--hf-dir", required=True, help="Path to Hugging Face VoxCPM2 snapshot")
    parser.add_argument("--out", required=True, help="Output GGUF path")
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16", "q8_0", "q4_k_m"])
    parser.add_argument("--emit-shape-manifest", default=None)
    args = parser.parse_args()

    hf_dir = Path(args.hf_dir)
    cfg_path = hf_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"missing config.json: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    print("VoxCPM2 converter skeleton")
    print(f"architecture={cfg.get('architecture')}")
    print(f"out={args.out}")
    print("TODO: implement GGUF writer and tensor conversion. See model_conversion.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
