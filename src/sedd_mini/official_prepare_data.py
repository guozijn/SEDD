from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .data import TOY_PRETRAIN_TEXTS, TOY_SFT_RECORDS, read_sft_records, read_text_records


@dataclass
class OfficialEncodedDataset:
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    metadata: dict[str, Any]


def get_gpt2_tokenizer():
    try:
        from transformers import GPT2TokenizerFast
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Official data preparation requires transformers. Run `uv sync --extra official`."
        ) from exc
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    except Exception:  # noqa: BLE001
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_pretrain_dataset(
    texts: list[str],
    *,
    seq_len: int,
    seed: int,
) -> OfficialEncodedDataset:
    tokenizer = get_gpt2_tokenizer()
    eos = int(tokenizer.eos_token_id)
    stream: list[int] = []
    for text in texts or TOY_PRETRAIN_TEXTS:
        stream.extend(tokenizer(text).input_ids + [eos])
    chunks = [stream[i : i + seq_len] for i in range(0, len(stream), seq_len)]
    random.Random(seed).shuffle(chunks)
    rows: list[list[int]] = []
    masks: list[list[bool]] = []
    for chunk in chunks:
        if len(chunk) < 2:
            continue
        padded = chunk[:seq_len] + [eos] * max(0, seq_len - len(chunk))
        rows.append(padded)
        masks.append([idx < min(len(chunk), seq_len) for idx in range(seq_len)])
    return OfficialEncodedDataset(
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(masks, dtype=torch.bool),
        {"mode": "official_pretrain", "seq_len": seq_len, "num_texts": len(texts)},
    )


def build_sft_dataset(records: list[dict[str, str]], *, seq_len: int) -> OfficialEncodedDataset:
    tokenizer = get_gpt2_tokenizer()
    eos = int(tokenizer.eos_token_id)
    rows: list[list[int]] = []
    masks: list[list[bool]] = []
    for record in records or TOY_SFT_RECORDS:
        prefix = f"User: {record['prompt']}\nAssistant: "
        prefix_ids = tokenizer(prefix).input_ids
        response_ids = tokenizer(record["response"]).input_ids + [eos]
        ids = (prefix_ids + response_ids)[:seq_len]
        response_start = min(len(prefix_ids), seq_len)
        padded = ids + [eos] * max(0, seq_len - len(ids))
        loss_mask = [False] * seq_len
        for pos in range(response_start, min(len(ids), seq_len)):
            loss_mask[pos] = True
        if any(loss_mask):
            rows.append(padded)
            masks.append(loss_mask)
    return OfficialEncodedDataset(
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(masks, dtype=torch.bool),
        {"mode": "official_sft", "seq_len": seq_len, "num_records": len(records)},
    )


def read_hf_sft_records(
    dataset_name: str,
    *,
    split: str,
    limit: int | None,
    response_mode: str,
) -> list[dict[str, str]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Hugging Face loading requires `uv sync --extra datasets`.") from exc
    rows = load_dataset(dataset_name, split=split)
    records: list[dict[str, str]] = []
    for row in rows:
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        solution = str(row.get("solution") or "").strip()
        deepseek_trace = str(row.get("deepseek_thinking_trajectory") or "").strip()
        deepseek_attempt = str(row.get("deepseek_attempt") or "").strip()
        if response_mode == "solution":
            response = solution
        elif response_mode == "deepseek":
            response = deepseek_trace or deepseek_attempt or solution
        elif response_mode == "deepseek_with_answer":
            parts = [part for part in [deepseek_trace, deepseek_attempt or solution] if part]
            response = "\n\nFinal answer:\n".join(parts)
        else:
            raise ValueError(f"unknown response mode: {response_mode}")
        if not response:
            continue
        records.append({"prompt": question, "response": response})
        if limit and len(records) >= limit:
            break
    return records


def split_dataset(
    dataset: OfficialEncodedDataset, valid_ratio: float, seed: int
) -> tuple[OfficialEncodedDataset, OfficialEncodedDataset]:
    n = int(dataset.input_ids.shape[0])
    if n == 0:
        raise ValueError("official dataset is empty")
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    valid_n = max(1, int(n * valid_ratio)) if n > 1 else 1
    valid_idx = torch.tensor(indices[:valid_n], dtype=torch.long)
    train_idx = torch.tensor(indices[valid_n:] or indices[:valid_n], dtype=torch.long)

    def take(idx: torch.Tensor, split: str) -> OfficialEncodedDataset:
        metadata = dict(dataset.metadata)
        metadata["split"] = split
        return OfficialEncodedDataset(dataset.input_ids[idx], dataset.loss_mask[idx], metadata)

    return take(train_idx, "train"), take(valid_idx, "valid")


def save_dataset(dataset: OfficialEncodedDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": dataset.input_ids,
            "loss_mask": dataset.loss_mask,
            "metadata": dataset.metadata,
        },
        path,
    )


def write_toy_sft_jsonl(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in TOY_SFT_RECORDS:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare GPT-2-tokenized data for official SEDD.")
    parser.add_argument("--mode", choices=["pretrain", "sft"], default="sft")
    parser.add_argument("--input", default="", help="Text path or SFT JSONL path. Empty uses toy data.")
    parser.add_argument("--hf-dataset", default="", help="Optional HF dataset name.")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument(
        "--response-mode",
        choices=["solution", "deepseek", "deepseek_with_answer"],
        default="solution",
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--name", default="", help="Prefix. Defaults to official_<mode>.")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--response-field", default="response")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input or None
    limit = args.limit or None
    if args.mode == "pretrain":
        texts = read_text_records(input_path, text_field=args.text_field, limit=limit)
        dataset = build_pretrain_dataset(texts, seq_len=args.seq_len, seed=args.seed)
    else:
        if args.hf_dataset:
            records = read_hf_sft_records(
                args.hf_dataset,
                split=args.hf_split,
                limit=limit,
                response_mode=args.response_mode,
            )
        else:
            records = read_sft_records(
                input_path,
                prompt_field=args.prompt_field,
                response_field=args.response_field,
                limit=limit,
            )
        dataset = build_sft_dataset(records, seq_len=args.seq_len)
    train, valid = split_dataset(dataset, args.valid_ratio, args.seed)
    prefix = args.name or f"official_{args.mode}"
    output_dir = Path(args.output_dir)
    train_path = output_dir / f"{prefix}_train.pt"
    valid_path = output_dir / f"{prefix}_valid.pt"
    save_dataset(train, train_path)
    save_dataset(valid, valid_path)
    print(f"wrote {train_path} rows={len(train.input_ids)}")
    print(f"wrote {valid_path} rows={len(valid.input_ids)}")


if __name__ == "__main__":
    main()
