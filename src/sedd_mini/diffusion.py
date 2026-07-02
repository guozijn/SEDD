from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .utils import safe_exp


@dataclass
class DiffusionBatch:
    perturbed: torch.Tensor
    masked_positions: torch.Tensor
    sigma: torch.Tensor
    dsigma: torch.Tensor


def loglinear_noise(t: torch.Tensor, eps: float = 1.0e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """Noise schedule from the SEDD absorbing graph setup."""

    t = t.reshape(-1).clamp(eps, 1.0 - eps)
    one_minus = 1.0 - (1.0 - eps) * t
    sigma = -torch.log(one_minus)
    dsigma = (1.0 - eps) / one_minus
    return sigma, dsigma


def sample_time(batch_size: int, device: torch.device, eps: float = 1.0e-3) -> torch.Tensor:
    return torch.rand(batch_size, device=device) * (1.0 - 2.0 * eps) + eps


def perturb_absorbing(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    mask_id: int,
    eps: float = 1.0e-3,
    t: torch.Tensor | None = None,
) -> DiffusionBatch:
    batch_size = input_ids.shape[0]
    if t is None:
        t = sample_time(batch_size, input_ids.device, eps)
    sigma, dsigma = loglinear_noise(t, eps)
    move_prob = 1.0 - torch.exp(-sigma)
    random_mask = torch.rand(input_ids.shape, device=input_ids.device) < move_prob[:, None]
    masked_positions = random_mask & loss_mask.bool()
    perturbed = torch.where(masked_positions, torch.full_like(input_ids, mask_id), input_ids)
    return DiffusionBatch(perturbed=perturbed, masked_positions=masked_positions, sigma=sigma, dsigma=dsigma)


def score_entropy_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    mask_id: int,
    eps: float = 1.0e-3,
    t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Denoising score entropy for an absorbing discrete diffusion process.

    This mirrors the absorbing-graph loss from Lou, Meng, and Ermon. Only positions
    that jumped into the absorbing mask state contribute to the objective.
    """

    diff = perturb_absorbing(input_ids, loss_mask, mask_id=mask_id, eps=eps, t=t)
    log_score = model(diff.perturbed, diff.sigma)
    score_data = log_score[..., :mask_id]

    active = diff.masked_positions & loss_mask.bool()
    target = input_ids.clamp(min=0, max=mask_id - 1)
    target_log_score = torch.gather(score_data, dim=-1, index=target[..., None]).squeeze(-1)

    ratio = 1.0 / torch.expm1(diff.sigma).clamp_min(1.0e-8)
    ratio = ratio[:, None]
    pos_term = safe_exp(score_data).sum(dim=-1)
    const = ratio * (torch.log(ratio.clamp_min(1.0e-8)) - 1.0)
    token_loss = pos_term - ratio * target_log_score + const
    weighted = token_loss * active.float() * diff.dsigma[:, None]
    denom = active.float().sum().clamp_min(1.0)
    loss = weighted.sum() / denom
    metrics = {
        "masked_fraction": float(active.float().mean().detach().cpu()),
        "mean_sigma": float(diff.sigma.mean().detach().cpu()),
        "active_tokens": float(active.float().sum().detach().cpu()),
    }
    return loss, metrics


@torch.no_grad()
def denoise_cross_entropy(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    mask_id: int,
    mask_rate: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = (torch.rand(input_ids.shape, device=input_ids.device) < mask_rate) & loss_mask.bool()
    perturbed = torch.where(active, torch.full_like(input_ids, mask_id), input_ids)
    sigma_value = -torch.log(torch.tensor(1.0 - mask_rate, device=input_ids.device).clamp_min(1.0e-5))
    sigma = sigma_value.expand(input_ids.shape[0])
    logits = model(perturbed, sigma)[..., :mask_id]
    labels = input_ids.clamp(min=0, max=mask_id - 1)
    loss = F.cross_entropy(logits[active], labels[active]) if active.any() else logits.sum() * 0.0
    pred = logits.argmax(dim=-1)
    acc = (pred[active] == labels[active]).float().mean() if active.any() else torch.tensor(0.0)
    return loss, acc
