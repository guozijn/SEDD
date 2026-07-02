from __future__ import annotations

import argparse
from pathlib import Path

from .data import (
    build_pretrain_dataset,
    build_sft_dataset,
    read_sft_records,
    read_text_records,
    save_dataset,
    split_dataset,
)
from .tokenizer import ByteTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare byte-tokenized SEDD datasets.")
    parser.add_argument("--mode", choices=["pretrain", "sft"], default="pretrain")
    parser.add_argument("--input", default="", help="Text directory/file or JSONL path. Empty uses toy data.")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--name", default="", help="Dataset name prefix. Defaults to mode.")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--response-field", default="response")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = ByteTokenizer()
    input_path = args.input or None
    limit = args.limit or None
    if args.mode == "pretrain":
        texts = read_text_records(input_path, text_field=args.text_field, limit=limit)
        dataset = build_pretrain_dataset(texts, tokenizer, seq_len=args.seq_len)
    else:
        records = read_sft_records(
            input_path,
            prompt_field=args.prompt_field,
            response_field=args.response_field,
            limit=limit,
        )
        dataset = build_sft_dataset(records, tokenizer, seq_len=args.seq_len)

    train, valid = split_dataset(dataset, args.valid_ratio, args.seed)
    output_dir = Path(args.output_dir)
    prefix = args.name or args.mode
    train_path = output_dir / f"{prefix}_train.pt"
    valid_path = output_dir / f"{prefix}_valid.pt"
    save_dataset(train, train_path)
    save_dataset(valid, valid_path)
    print(f"wrote {train_path} rows={len(train.input_ids)}")
    print(f"wrote {valid_path} rows={len(valid.input_ids)}")


if __name__ == "__main__":
    main()
