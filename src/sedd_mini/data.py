from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer


TOY_PRETRAIN_TEXTS = [
    "Score entropy discrete diffusion learns ratios p_t(y) / p_t(x), not next-token logits.",
    "A masked absorbing process is a simple CTMC for text: clean tokens jump into a mask state.",
    "At sampling time the model repeatedly replaces mask tokens with likely clean tokens.",
    "Bidirectional attention makes infilling and non-left-to-right generation natural.",
    "For supervised fine-tuning, keep the prompt visible and apply the loss only to answer tokens.",
    "For reinforcement learning, treat denoising choices as policy actions and optimize reward-weighted log probabilities.",
]

TOY_SFT_RECORDS = [
    {
        "prompt": "Explain SEDD in one sentence.",
        "response": "SEDD is a discrete diffusion language model that learns score ratios with a score-entropy objective.",
    },
    {
        "prompt": "Why is the absorbing mask useful?",
        "response": "It gives a simple forward corruption process and makes prompt-conditioned denoising easy to implement.",
    },
    {
        "prompt": "How would you adapt RL to SEDD?",
        "response": "Sample a denoising trajectory, score the completed text with a reward, and update the trajectory log probabilities with a KL penalty to a reference model.",
    },
]


@dataclass
class EncodedDataset:
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    metadata: dict[str, Any]


class TokenDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu")
        self.input_ids = payload["input_ids"].long()
        self.loss_mask = payload["loss_mask"].bool()
        self.metadata = payload.get("metadata", {})

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[idx], "loss_mask": self.loss_mask[idx]}


def collate_batch(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([row["input_ids"] for row in rows], dim=0),
        "loss_mask": torch.stack([row["loss_mask"] for row in rows], dim=0),
    }


def read_text_records(
    path: str | Path | None,
    *,
    text_field: str = "text",
    limit: int | None = None,
) -> list[str]:
    if path is None:
        return list(TOY_PRETRAIN_TEXTS)
    path = Path(path)
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    texts: list[str] = []
    for file in files:
        if not file.is_file():
            continue
        if file.suffix == ".jsonl":
            with file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        value = obj.get(text_field)
                        if value:
                            texts.append(str(value))
                    if limit and len(texts) >= limit:
                        return texts
        else:
            text = file.read_text(encoding="utf-8", errors="replace")
            if file.suffix in {".txt", ".md", ".text", ""}:
                texts.extend(chunk.strip() for chunk in text.split("\n\n") if chunk.strip())
            else:
                texts.append(text)
        if limit and len(texts) >= limit:
            return texts[:limit]
    return texts[:limit] if limit else texts


def read_sft_records(
    path: str | Path | None,
    *,
    prompt_field: str = "prompt",
    response_field: str = "response",
    limit: int | None = None,
) -> list[dict[str, str]]:
    if path is None:
        return list(TOY_SFT_RECORDS)
    records: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get(prompt_field) or obj.get("instruction") or obj.get("question")
            response = obj.get(response_field) or obj.get("output") or obj.get("answer")
            if prompt is None or response is None:
                continue
            records.append({"prompt": str(prompt), "response": str(response)})
            if limit and len(records) >= limit:
                break
    return records


def build_pretrain_dataset(
    texts: list[str],
    tokenizer: ByteTokenizer,
    *,
    seq_len: int,
    shuffle: bool = True,
) -> EncodedDataset:
    stream: list[int] = []
    for text in texts:
        stream.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    if shuffle:
        chunks = [stream[i : i + seq_len] for i in range(0, len(stream), seq_len)]
        random.shuffle(chunks)
    else:
        chunks = [stream[i : i + seq_len] for i in range(0, len(stream), seq_len)]
    input_rows: list[list[int]] = []
    mask_rows: list[list[bool]] = []
    for chunk in chunks:
        if len(chunk) < 4:
            continue
        padded = chunk[:seq_len] + [tokenizer.pad_id] * max(0, seq_len - len(chunk))
        input_rows.append(padded)
        mask_rows.append([token != tokenizer.pad_id for token in padded])
    return EncodedDataset(
        input_ids=torch.tensor(input_rows, dtype=torch.long),
        loss_mask=torch.tensor(mask_rows, dtype=torch.bool),
        metadata={"mode": "pretrain", "num_texts": len(texts), "seq_len": seq_len},
    )


def build_sft_dataset(
    records: list[dict[str, str]],
    tokenizer: ByteTokenizer,
    *,
    seq_len: int,
) -> EncodedDataset:
    input_rows: list[list[int]] = []
    mask_rows: list[list[bool]] = []
    for row in records:
        prefix = f"User: {row['prompt']}\nAssistant: "
        prefix_ids = tokenizer.encode(prefix, add_bos=True, add_eos=False)
        response_ids = tokenizer.encode(row["response"], add_bos=False, add_eos=True)
        ids = (prefix_ids + response_ids)[:seq_len]
        response_start = min(len(prefix_ids), seq_len)
        padded = ids + [tokenizer.pad_id] * max(0, seq_len - len(ids))
        loss_mask = [False] * seq_len
        for pos in range(response_start, min(len(ids), seq_len)):
            loss_mask[pos] = padded[pos] != tokenizer.pad_id
        if not any(loss_mask):
            continue
        input_rows.append(padded)
        mask_rows.append(loss_mask)
    return EncodedDataset(
        input_ids=torch.tensor(input_rows, dtype=torch.long),
        loss_mask=torch.tensor(mask_rows, dtype=torch.bool),
        metadata={"mode": "sft", "num_records": len(records), "seq_len": seq_len},
    )


def split_dataset(dataset: EncodedDataset, valid_ratio: float, seed: int) -> tuple[EncodedDataset, EncodedDataset]:
    n = int(dataset.input_ids.shape[0])
    if n == 0:
        raise ValueError("dataset is empty")
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    valid_n = max(1, int(n * valid_ratio)) if n > 1 else 1
    valid_idx = torch.tensor(indices[:valid_n], dtype=torch.long)
    train_idx = torch.tensor(indices[valid_n:] or indices[:valid_n], dtype=torch.long)

    def take(idx: torch.Tensor, split: str) -> EncodedDataset:
        metadata = dict(dataset.metadata)
        metadata["split"] = split
        return EncodedDataset(dataset.input_ids[idx], dataset.loss_mask[idx], metadata)

    return take(train_idx, "train"), take(valid_idx, "valid")


def save_dataset(dataset: EncodedDataset, path: str | Path) -> None:
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
