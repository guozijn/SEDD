from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import save_config
from .model import build_model
from .utils import ExponentialMovingAverage


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    ema: ExponentialMovingAverage | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "config": config,
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    save_config(config, path.with_suffix(".yaml"))


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    config = payload["config"]
    model = build_model(config["model"]).to(device)
    model.load_state_dict(payload["model"])
    if use_ema and "ema" in payload:
        ema = ExponentialMovingAverage(model.parameters(), decay=payload["ema"]["decay"])
        ema.load_state_dict(payload["ema"])
        ema.copy_to(model.parameters())
    model.eval()
    return model, config, payload
