from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .config import load_config, parse_key_value_overrides, save_config
from .data import TokenDataset, collate_batch
from .diffusion import score_entropy_loss
from .model import build_model
from .utils import (
    ExponentialMovingAverage,
    count_parameters,
    cycle,
    get_device,
    json_log,
    learning_rate,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or supervised-fine-tune SEDD-mini.")
    parser.add_argument("--config", default="", help="YAML config path.")
    parser.add_argument("overrides", nargs="*", help="Dotted overrides, e.g. train.steps=100")
    return parser


def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    mask_id: int,
    eps: float,
    max_batches: int = 20,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            loss, _ = score_entropy_loss(model, input_ids, loss_mask, mask_id=mask_id, eps=eps)
            losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    args = build_parser().parse_args()
    overrides = parse_key_value_overrides(args.overrides)
    cfg = load_config(args.config or None, overrides)
    train_cfg: dict[str, Any] = cfg["train"]
    set_seed(int(train_cfg["seed"]))
    device = get_device(str(train_cfg["device"]))

    out_dir = Path(train_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    train_data = TokenDataset(train_cfg["train_path"])
    valid_data = TokenDataset(train_cfg["valid_path"])
    train_loader = DataLoader(
        train_data,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=collate_batch,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=collate_batch,
        drop_last=False,
    )
    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    ema = ExponentialMovingAverage(model.parameters(), decay=float(train_cfg["ema_decay"]))
    start_step = 0
    if train_cfg.get("resume"):
        payload = torch.load(train_cfg["resume"], map_location=device)
        model.load_state_dict(payload["model"])
        payload_stage = payload.get("config", {}).get("train", {}).get("stage")
        same_stage = payload_stage == train_cfg["stage"]
        if same_stage and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if same_stage and "ema" in payload:
            ema.load_state_dict(payload["ema"])
        start_step = int(payload.get("step", 0)) if same_stage else 0
        json_log(
            {
                "event": "resume",
                "path": train_cfg["resume"],
                "payload_stage": payload_stage,
                "current_stage": train_cfg["stage"],
                "continued_optimizer": same_stage,
                "start_step": start_step,
            }
        )

    use_amp = bool(train_cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    iterator = cycle(train_loader)
    model.train()
    t0 = time.time()

    json_log(
        {
            "event": "train_start",
            "device": str(device),
            "params": count_parameters(model),
            "stage": train_cfg["stage"],
            "train_rows": len(train_data),
            "valid_rows": len(valid_data),
        }
    )

    for step in tqdm(range(start_step + 1, int(train_cfg["steps"]) + 1), initial=start_step):
        batch = next(iterator)
        input_ids = batch["input_ids"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        lr = learning_rate(step, float(train_cfg["lr"]), int(train_cfg["warmup_steps"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss, metrics = score_entropy_loss(
                model,
                input_ids,
                loss_mask,
                mask_id=int(cfg["model"]["mask_id"]),
                eps=float(train_cfg["sampling_eps"]),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        ema.update(model.parameters())

        if step % int(train_cfg["log_every"]) == 0:
            payload = {
                "event": "train_step",
                "step": step,
                "loss": float(loss.detach().cpu()),
                "lr": lr,
                "elapsed_s": round(time.time() - t0, 2),
            }
            payload.update(metrics)
            json_log(payload)
        if step % int(train_cfg["eval_every"]) == 0:
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            valid_loss = evaluate_loss(
                model,
                valid_loader,
                device=device,
                mask_id=int(cfg["model"]["mask_id"]),
                eps=float(train_cfg["sampling_eps"]),
            )
            ema.restore(model.parameters())
            json_log({"event": "eval", "step": step, "valid_loss": valid_loss})
        if step % int(train_cfg["save_every"]) == 0:
            save_checkpoint(
                out_dir / f"checkpoint_{step}.pt",
                model=model,
                config=cfg,
                step=step,
                optimizer=optimizer,
                ema=ema,
            )

    save_checkpoint(
        out_dir / "checkpoint_last.pt",
        model=model,
        config=cfg,
        step=int(train_cfg["steps"]),
        optimizer=optimizer,
        ema=ema,
    )
    json_log({"event": "train_done", "checkpoint": str(out_dir / "checkpoint_last.pt")})


if __name__ == "__main__":
    main()
