#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.pes2o_dedup import DedupConfig, prepare_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Raw, MinHashLSH, and LSHBloom peS2o variants."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-documents", required=True, type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-perm", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--effective-fp", type=float, default=1e-10)
    parser.add_argument("--training-gate-rate", type=float, default=0.05)
    parser.add_argument(
        "--tokenizer",
        help="Optional Hugging Face tokenizer name, e.g. Qwen/Qwen2.5-0.5B.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        print(f"Loading tokenizer: {args.tokenizer}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    config = DedupConfig(
        expected_documents=args.expected_documents,
        threshold=args.threshold,
        num_perm=args.num_perm,
        seed=args.seed,
        effective_fp=args.effective_fp,
        training_gate_rate=args.training_gate_rate,
    )

    def show_progress(documents: int) -> None:
        print(f"Processed {documents:,}/{config.expected_documents:,} documents", flush=True)

    manifest = prepare_variants(
        input_path=args.input,
        output_dir=args.output_dir,
        config=config,
        tokenizer=tokenizer,
        progress_callback=show_progress,
    )
    print(json.dumps(manifest["training_gate"], indent=2), flush=True)
    print(f"Manifest: {args.output_dir / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
