from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "seq_len": 128,
        "vocab_size": 260,
        "mask_id": 259,
        "pad_id": 0,
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 4,
        "d_ff": 1024,
        "dropout": 0.1,
    },
    "train": {
        "stage": "pretrain",
        "train_path": "data/processed/pretrain_train.pt",
        "valid_path": "data/processed/pretrain_valid.pt",
        "out_dir": "runs/pretrain",
        "resume": "",
        "batch_size": 16,
        "num_workers": 0,
        "steps": 1000,
        "lr": 3.0e-4,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "grad_clip": 1.0,
        "eval_every": 100,
        "save_every": 500,
        "log_every": 10,
        "ema_decay": 0.999,
        "amp": True,
        "seed": 13,
        "device": "auto",
        "sampling_eps": 1.0e-3,
    },
    "sampling": {
        "steps": 32,
        "max_new_tokens": 96,
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 0.95,
    },
    "rl": {
        "prompts_path": "data/processed/rl_prompts.txt",
        "checkpoint": "runs/sft/checkpoint_last.pt",
        "reference_checkpoint": "",
        "out_dir": "runs/rl",
        "updates": 100,
        "batch_size": 4,
        "lr": 1.0e-5,
        "max_new_tokens": 96,
        "sample_steps": 16,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
        "kl_coef": 0.02,
        "entropy_coef": 0.0,
        "reward": {
            "kind": "keyword_length",
            "keyword": "because",
            "keyword_bonus": 1.0,
            "min_chars": 80,
            "max_chars": 500,
            "length_bonus": 0.5,
        },
        "seed": 17,
        "device": "auto",
        "save_every": 25,
        "log_every": 1,
    },
}


def recursive_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            recursive_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path:
        with Path(path).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        recursive_update(cfg, loaded)
    if overrides:
        recursive_update(cfg, overrides)
    return cfg


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def parse_key_value_overrides(items: list[str]) -> dict[str, Any]:
    """Parse CLI overrides in dotted form, e.g. train.steps=10."""

    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor = result
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return result
