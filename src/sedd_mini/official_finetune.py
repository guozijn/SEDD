from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import TokenDataset, collate_batch
from .official_backend import load_official_components, save_official_checkpoint
from .utils import cycle, get_device, json_log, learning_rate, set_seed


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.to(device=base.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return self.base(x) + update.to(dtype=self.base.weight.dtype)

    def merged_weight(self) -> torch.Tensor:
        update = self.lora_b.weight @ self.lora_a.weight
        return self.base.weight.detach().float() + update.detach().float() * self.scaling


def _target_matches(name: str, targets: list[str]) -> bool:
    return any(name == target or name.endswith(f".{target}") for target in targets)


def apply_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    targets: list[str],
) -> list[str]:
    for param in model.parameters():
        param.requires_grad_(False)

    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if name.startswith("sigma_map."):
            continue
        if not isinstance(module, nn.Linear) or not _target_matches(name, targets):
            continue
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(name)
    if not replaced:
        raise ValueError(f"No Linear modules matched LoRA targets: {targets}")
    return replaced


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def total_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def merged_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    lora_modules = {
        name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)
    }
    state: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if any(key.startswith(f"{prefix}.") for prefix in lora_modules):
            continue
        state[key] = value.detach().cpu()
    for prefix, module in lora_modules.items():
        state[f"{prefix}.weight"] = module.merged_weight().cpu()
        if module.base.bias is not None:
            state[f"{prefix}.bias"] = module.base.bias.detach().cpu()
    return state


def save_merged_lora_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    base_model_path: str,
    step: int,
    optimizer: torch.optim.Optimizer | None,
    extra: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "backend": "official",
        "base_model_path": base_model_path,
        "model": merged_lora_state_dict(model),
        "step": step,
        "extra": extra,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def official_score_entropy_loss(
    model: torch.nn.Module,
    graph: Any,
    noise: torch.nn.Module,
    clean: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eps: float = 1.0e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size = clean.shape[0]
    t = torch.rand(batch_size, device=clean.device) * (1.0 - 2.0 * eps) + eps
    sigma, dsigma = noise(t)
    perturbed_all = graph.sample_transition(clean, sigma[:, None])
    loss_mask = loss_mask.bool()
    perturbed = torch.where(loss_mask, perturbed_all, clean)
    log_score = model(perturbed, sigma)
    token_loss = graph.score_entropy(log_score, sigma[:, None], perturbed, clean)
    weighted = token_loss * loss_mask.float() * dsigma[:, None]
    denom = loss_mask.float().sum().clamp_min(1.0)
    loss = weighted.sum() / denom
    active = (perturbed != clean) & loss_mask
    return loss, {
        "mean_sigma": float(sigma.mean().detach().cpu()),
        "active_tokens": float(active.float().sum().detach().cpu()),
        "eligible_tokens": float(loss_mask.float().sum().detach().cpu()),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    graph: Any,
    noise: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        clean = batch["input_ids"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        loss, _ = official_score_entropy_loss(model, graph, noise, clean, loss_mask)
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT/fine-tune official SEDD checkpoints.")
    parser.add_argument("--model-path", default="runs/arc_models/base/checkpoint_base.pt")
    parser.add_argument("--official-repo", default="external/Score-Entropy-Discrete-Diffusion")
    parser.add_argument("--train-path", default="data/processed/official_arc_easy_train.pt")
    parser.add_argument("--valid-path", default="data/processed/official_arc_easy_valid.pt")
    parser.add_argument("--out-dir", default="runs/arc_models/arc_lora_sft")
    parser.add_argument("--resume", default="", help="Optional official fine-tune checkpoint.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Intermediate checkpoint interval. Set 0 to keep only checkpoint_last.pt.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-eval-batches", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-targets",
        default="attn_qkv,attn_out,mlp.0,mlp.2",
        help="Comma-separated Linear module suffixes to adapt. Set rank=0 for full fine-tune.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=31)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_path = args.resume or args.model_path
    model, graph, noise, base_model_path, loaded_step = load_official_components(
        load_path,
        repo_path=args.official_repo,
        device=device,
    )
    lora_targets = [target.strip() for target in args.lora_targets.split(",") if target.strip()]
    lora_modules: list[str] = []
    if args.lora_rank > 0:
        lora_modules = apply_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            targets=lora_targets,
        )
    model.train()
    train_data = TokenDataset(args.train_path)
    valid_data = TokenDataset(args.valid_path)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        drop_last=False,
    )
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume:
        payload = torch.load(args.resume, map_location=device)
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])

    iterator = cycle(train_loader)
    start_time = time.time()
    json_log(
        {
            "event": "official_sft_start",
            "device": str(device),
            "model_path": args.model_path,
            "base_model_path": base_model_path,
            "resume": args.resume,
            "loaded_step": loaded_step,
            "train_rows": len(train_data),
            "valid_rows": len(valid_data),
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_modules": lora_modules,
            "trainable_parameters": trainable_parameter_count(model),
            "total_parameters": total_parameter_count(model),
        }
    )

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    accum_count = 0
    for step in tqdm(range(1, args.steps + 1)):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.lr, args.warmup_steps)
        batch = next(iterator)
        clean = batch["input_ids"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        loss, metrics = official_score_entropy_loss(model, graph, noise, clean, loss_mask)
        scaled_loss = loss / max(args.grad_accum, 1)
        scaled_loss.backward()
        accum_loss += float(loss.detach().cpu())
        accum_count += 1
        if step % args.grad_accum == 0 or step == args.steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if args.log_every > 0 and (step % args.log_every == 0 or step == args.steps):
            payload = {
                "event": "official_sft_step",
                "step": step,
                "loss": accum_loss / max(accum_count, 1),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": round(time.time() - start_time, 2),
            }
            payload.update(metrics)
            json_log(payload)
            accum_loss = 0.0
            accum_count = 0
        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
            valid_loss = evaluate(
                model,
                graph,
                noise,
                valid_loader,
                device=device,
                max_batches=args.max_eval_batches,
            )
            json_log({"event": "official_sft_eval", "step": step, "valid_loss": valid_loss})
        if args.save_every > 0 and step % args.save_every == 0:
            if args.lora_rank > 0:
                save_merged_lora_checkpoint(
                    out_dir / f"checkpoint_{step}.pt",
                    model=model,
                    base_model_path=base_model_path,
                    step=step,
                    optimizer=optimizer,
                    extra={
                        "source_model_path": args.model_path,
                        "stage": "arc_lora_sft",
                        "lora_rank": args.lora_rank,
                        "lora_alpha": args.lora_alpha,
                        "lora_dropout": args.lora_dropout,
                        "lora_targets": lora_targets,
                    },
                )
            else:
                save_official_checkpoint(
                    out_dir / f"checkpoint_{step}.pt",
                    model=model,
                    base_model_path=base_model_path,
                    step=step,
                    optimizer=optimizer,
                    extra={"source_model_path": args.model_path, "stage": "sft"},
                )

    if args.lora_rank > 0:
        save_merged_lora_checkpoint(
            out_dir / "checkpoint_last.pt",
            model=model,
            base_model_path=base_model_path,
            step=args.steps,
            optimizer=optimizer,
            extra={
                "source_model_path": args.model_path,
                "stage": "arc_lora_sft",
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_targets": lora_targets,
            },
        )
    else:
        save_official_checkpoint(
            out_dir / "checkpoint_last.pt",
            model=model,
            base_model_path=base_model_path,
            step=args.steps,
            optimizer=optimizer,
            extra={"source_model_path": args.model_path, "stage": "sft"},
        )
    json_log({"event": "official_sft_done", "checkpoint": str(out_dir / "checkpoint_last.pt")})


if __name__ == "__main__":
    main()
