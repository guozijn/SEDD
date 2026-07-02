from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .data import TokenDataset, collate_batch
from .diffusion import denoise_cross_entropy, score_entropy_loss
from .utils import get_device, json_log, perplexity_from_xent, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SEDD-mini checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=19)
    return parser


@torch.no_grad()
def pseudo_perplexity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    mask_id: int,
) -> tuple[float, float]:
    masked = torch.where(loss_mask, torch.full_like(input_ids, mask_id), input_ids)
    sigma = torch.ones(input_ids.shape[0], device=input_ids.device)
    logits = model(masked, sigma)[..., :mask_id]
    labels = input_ids.clamp(min=0, max=mask_id - 1)
    active_logits = logits[loss_mask]
    active_labels = labels[loss_mask]
    if active_labels.numel() == 0:
        return 0.0, 0.0
    xent = F.cross_entropy(active_logits, active_labels).item()
    acc = (active_logits.argmax(dim=-1) == active_labels).float().mean().item()
    return xent, acc


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device=device, use_ema=True)
    dataset = TokenDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    se_losses: list[float] = []
    ce_losses: list[float] = []
    ce_accs: list[float] = []
    pseudo_xents: list[float] = []
    pseudo_accs: list[float] = []
    for idx, batch in enumerate(loader):
        if idx >= args.max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        se_loss, _ = score_entropy_loss(
            model,
            input_ids,
            loss_mask,
            mask_id=int(cfg["model"]["mask_id"]),
            eps=float(cfg["train"]["sampling_eps"]),
        )
        ce_loss, ce_acc = denoise_cross_entropy(
            model,
            input_ids,
            loss_mask,
            mask_id=int(cfg["model"]["mask_id"]),
        )
        xent, pseudo_acc = pseudo_perplexity(
            model, input_ids, loss_mask, mask_id=int(cfg["model"]["mask_id"])
        )
        se_losses.append(float(se_loss.detach().cpu()))
        ce_losses.append(float(ce_loss.detach().cpu()))
        ce_accs.append(float(ce_acc.detach().cpu()))
        pseudo_xents.append(xent)
        pseudo_accs.append(pseudo_acc)

    def mean(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    json_log(
        {
            "event": "eval_done",
            "checkpoint": args.checkpoint,
            "data": args.data,
            "score_entropy": mean(se_losses),
            "denoise_ce": mean(ce_losses),
            "denoise_acc": mean(ce_accs),
            "pseudo_xent": mean(pseudo_xents),
            "pseudo_ppl": perplexity_from_xent(mean(pseudo_xents)),
            "pseudo_acc": mean(pseudo_accs),
            "batches": len(se_losses),
        }
    )


if __name__ == "__main__":
    main()
