from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def json_log(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def cycle(iterable: Iterable[Any]):
    while True:
        for item in iterable:
            yield item


class ExponentialMovingAverage:
    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay: float) -> None:
        self.decay = decay
        self.shadow = [p.detach().clone() for p in parameters if p.requires_grad]
        self.backup: list[torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        trainable = [p for p in parameters if p.requires_grad]
        for shadow, param in zip(self.shadow, trainable, strict=True):
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        trainable = [p for p in parameters if p.requires_grad]
        for shadow, param in zip(self.shadow, trainable, strict=True):
            param.copy_(shadow)

    @torch.no_grad()
    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        self.backup = [p.detach().clone() for p in parameters if p.requires_grad]

    @torch.no_grad()
    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        if self.backup is None:
            return
        trainable = [p for p in parameters if p.requires_grad]
        for backup, param in zip(self.backup, trainable, strict=True):
            param.copy_(backup)
        self.backup = None

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = [tensor.clone() for tensor in state["shadow"]]


def learning_rate(step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, step / warmup_steps)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def file_size_mb(path: str | Path) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def safe_exp(x: torch.Tensor, max_value: float = 20.0) -> torch.Tensor:
    return torch.exp(torch.clamp(x, max=max_value))


def perplexity_from_xent(xent: float) -> float:
    if not math.isfinite(xent):
        return float("inf")
    return float(math.exp(min(20.0, xent)))
