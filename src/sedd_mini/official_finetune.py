from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import TokenDataset, collate_batch
from .official_backend import load_official_components, save_official_checkpoint
from .utils import cycle, get_device, json_log, learning_rate, set_seed


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
    parser.add_argument("--model-path", default="louaaron/sedd-small")
    parser.add_argument("--official-repo", default="external/Score-Entropy-Discrete-Diffusion")
    parser.add_argument("--train-path", default="data/processed/official_sft_train.pt")
    parser.add_argument("--valid-path", default="data/processed/official_sft_valid.pt")
    parser.add_argument("--out-dir", default="runs/official_sft")
    parser.add_argument("--resume", default="", help="Optional official fine-tune checkpoint.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--max-eval-batches", type=int, default=10)
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
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
        }
    )

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
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
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            payload = {
                "event": "official_sft_step",
                "step": step,
                "loss": accum_loss / args.log_every,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": round(time.time() - start_time, 2),
            }
            payload.update(metrics)
            json_log(payload)
            accum_loss = 0.0
        if step % args.eval_every == 0:
            valid_loss = evaluate(
                model,
                graph,
                noise,
                valid_loader,
                device=device,
                max_batches=args.max_eval_batches,
            )
            json_log({"event": "official_sft_eval", "step": step, "valid_loss": valid_loss})
        if step % args.save_every == 0:
            save_official_checkpoint(
                out_dir / f"checkpoint_{step}.pt",
                model=model,
                base_model_path=base_model_path,
                step=step,
                optimizer=optimizer,
                extra={"source_model_path": args.model_path, "stage": "sft"},
            )

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
