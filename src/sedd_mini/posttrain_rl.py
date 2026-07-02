from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .checkpoint import load_checkpoint, save_checkpoint
from .config import load_config, parse_key_value_overrides, save_config
from .reward import compute_reward
from .sampling import SampleTrace, sample_response
from .tokenizer import ByteTokenizer
from .utils import get_device, json_log, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL post-train SEDD-mini with denoising policy gradients.")
    parser.add_argument("--config", default="", help="YAML config path.")
    parser.add_argument("overrides", nargs="*", help="Dotted overrides, e.g. rl.updates=10")
    return parser


def load_prompts(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        return [
            "Explain why score entropy is useful for discrete diffusion.",
            "Describe one challenge when adapting RLHF to SEDD.",
            "Give a concise answer about bidirectional attention in diffusion LMs.",
        ]
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def main() -> None:
    args = build_parser().parse_args()
    overrides = parse_key_value_overrides(args.overrides)
    cfg = load_config(args.config or None, overrides)
    rl_cfg: dict[str, Any] = cfg["rl"]
    set_seed(int(rl_cfg["seed"]))
    device = get_device(str(rl_cfg["device"]))
    out_dir = Path(rl_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "rl_config.yaml")

    model, base_cfg, _ = load_checkpoint(rl_cfg["checkpoint"], device=device, use_ema=True)
    cfg["model"] = base_cfg["model"]
    tokenizer = ByteTokenizer()
    reference_path = rl_cfg.get("reference_checkpoint") or rl_cfg["checkpoint"]
    reference_model, _, _ = load_checkpoint(reference_path, device=device, use_ema=True)
    for param in reference_model.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(rl_cfg["lr"]))
    prompts = load_prompts(rl_cfg["prompts_path"])
    baseline = 0.0
    beta = 0.9

    json_log(
        {
            "event": "rl_start",
            "device": str(device),
            "checkpoint": rl_cfg["checkpoint"],
            "reference": reference_path,
            "prompts": len(prompts),
        }
    )

    for update in tqdm(range(1, int(rl_cfg["updates"]) + 1)):
        optimizer.zero_grad(set_to_none=True)
        batch_rewards: list[float] = []
        batch_texts: list[str] = []
        policy_terms: list[torch.Tensor] = []
        kl_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []

        for _ in range(int(rl_cfg["batch_size"])):
            prompt = random.choice(prompts)
            trace = sample_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=int(rl_cfg["max_new_tokens"]),
                steps=int(rl_cfg["sample_steps"]),
                temperature=float(rl_cfg["temperature"]),
                top_k=int(rl_cfg["top_k"]),
                top_p=float(rl_cfg["top_p"]),
                seq_len=int(cfg["model"]["seq_len"]),
                device=device,
                reference_model=reference_model,
                return_trace=True,
            )
            assert isinstance(trace, SampleTrace)
            text = tokenizer.decode(trace.response_ids.tolist())
            reward = compute_reward(text, rl_cfg["reward"])
            batch_rewards.append(reward)
            batch_texts.append(text)
            advantage = reward - baseline
            policy_terms.append(-torch.tensor(advantage, device=device) * trace.logprob_sum)
            if trace.ref_logprob_sum is not None:
                kl_terms.append(trace.logprob_sum - trace.ref_logprob_sum)
            entropy_terms.append(trace.entropy_sum)

        mean_reward = sum(batch_rewards) / max(len(batch_rewards), 1)
        baseline = beta * baseline + (1.0 - beta) * mean_reward
        policy_loss = torch.stack(policy_terms).mean()
        kl_loss = torch.stack(kl_terms).mean() if kl_terms else torch.zeros((), device=device)
        entropy_bonus = torch.stack(entropy_terms).mean() if entropy_terms else torch.zeros((), device=device)
        loss = (
            policy_loss
            + float(rl_cfg["kl_coef"]) * kl_loss
            - float(rl_cfg["entropy_coef"]) * entropy_bonus
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if update % int(rl_cfg["log_every"]) == 0:
            json_log(
                {
                    "event": "rl_update",
                    "update": update,
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "kl_term": float(kl_loss.detach().cpu()),
                    "entropy": float(entropy_bonus.detach().cpu()),
                    "reward": mean_reward,
                    "baseline": baseline,
                    "sample": batch_texts[0][:240],
                }
            )
        if update % int(rl_cfg["save_every"]) == 0:
            save_checkpoint(
                out_dir / f"checkpoint_rl_{update}.pt",
                model=model,
                config=cfg,
                step=update,
                optimizer=optimizer,
                extra={"baseline": baseline, "rl_config": rl_cfg},
            )

    save_checkpoint(
        out_dir / "checkpoint_last.pt",
        model=model,
        config=cfg,
        step=int(rl_cfg["updates"]),
        optimizer=optimizer,
        extra={"baseline": baseline, "rl_config": rl_cfg},
    )
    json_log({"event": "rl_done", "checkpoint": str(out_dir / "checkpoint_last.pt")})


if __name__ == "__main__":
    main()
