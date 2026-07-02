from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .official_backend import load_official_components, save_official_checkpoint
from .reward import compute_reward
from .sampling import top_k_top_p_filter
from .utils import get_device, json_log, set_seed


@dataclass
class OfficialTrace:
    ids: torch.Tensor
    response_ids: torch.Tensor
    logprob_sum: torch.Tensor
    ref_logprob_sum: torch.Tensor
    entropy_sum: torch.Tensor


def load_prompts(path: str | Path) -> list[str]:
    path = Path(path)
    if path.exists():
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if prompts:
            return prompts
    return [
        "Explain score entropy for discrete diffusion.",
        "Describe how RL can be adapted to SEDD.",
    ]


def choose_positions(masked: torch.Tensor, step: int, steps: int) -> torch.Tensor:
    positions = masked.nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return positions
    remaining_steps = max(1, steps - step)
    count = max(1, int(torch.ceil(torch.tensor(positions.numel() / remaining_steps)).item()))
    return positions[:count]


def official_sample_trace(
    model: torch.nn.Module,
    reference_model: torch.nn.Module,
    noise: Any,
    tokenizer: Any,
    prompt: str,
    *,
    seq_len: int,
    mask_id: int,
    max_new_tokens: int,
    steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> OfficialTrace:
    eos = int(tokenizer.eos_token_id)
    prefix_ids = tokenizer(f"User: {prompt}\nAssistant: ").input_ids
    max_prompt = max(1, seq_len - max_new_tokens)
    prefix_ids = prefix_ids[-max_prompt:]
    gen_len = min(max_new_tokens, seq_len - len(prefix_ids))
    ids = torch.full((1, seq_len), eos, dtype=torch.long, device=device)
    ids[0, : len(prefix_ids)] = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    response_slice = slice(len(prefix_ids), len(prefix_ids) + gen_len)
    ids[0, response_slice] = mask_id
    response_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    response_mask[response_slice] = True
    logprob_sum = torch.zeros((), device=device)
    ref_logprob_sum = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)

    for step in range(steps):
        masked = (ids[0] == mask_id) & response_mask
        fill_positions = choose_positions(masked, step, steps)
        if fill_positions.numel() == 0:
            break
        t = torch.full((1,), 1.0 - (step / max(steps, 1)) * 0.999, device=device)
        sigma = noise(t)[0]
        logits = model(ids.clone(), sigma)[0, fill_positions, :mask_id]
        logits = top_k_top_p_filter(logits / max(temperature, 1.0e-5), top_k=top_k, top_p=top_p)
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        action_logprobs = torch.log(torch.gather(probs, 1, sampled[:, None]).squeeze(-1).clamp_min(1.0e-8))
        entropy = -(probs * torch.log(probs.clamp_min(1.0e-8))).sum(dim=-1)
        logprob_sum = logprob_sum + action_logprobs.sum()
        entropy_sum = entropy_sum + entropy.sum()
        with torch.no_grad():
            ref_logits = reference_model(ids.clone(), sigma)[0, fill_positions, :mask_id]
            ref_probs = torch.softmax(ref_logits / max(temperature, 1.0e-5), dim=-1)
            ref_logprobs = torch.log(
                torch.gather(ref_probs, 1, sampled[:, None]).squeeze(-1).clamp_min(1.0e-8)
            )
        ref_logprob_sum = ref_logprob_sum + ref_logprobs.sum()
        ids[0, fill_positions] = sampled

    ids[ids == mask_id] = eos
    return OfficialTrace(
        ids=ids.detach().clone(),
        response_ids=ids[0, response_slice].detach().clone(),
        logprob_sum=logprob_sum,
        ref_logprob_sum=ref_logprob_sum,
        entropy_sum=entropy_sum,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL post-train official SEDD checkpoints.")
    parser.add_argument("--model-path", default="runs/official_sft/checkpoint_last.pt")
    parser.add_argument("--reference-model-path", default="louaaron/sedd-small")
    parser.add_argument("--official-repo", default="external/Score-Entropy-Discrete-Diffusion")
    parser.add_argument("--prompts-path", default="data/processed/rl_prompts.txt")
    parser.add_argument("--out-dir", default="runs/official_rl")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5.0e-6)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--reward-kind", default="heuristic_helpfulness")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=37)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, graph, noise, base_model_path, loaded_step = load_official_components(
        args.model_path,
        repo_path=args.official_repo,
        device=device,
    )
    reference_model, _, _, _, _ = load_official_components(
        args.reference_model_path,
        repo_path=args.official_repo,
        device=device,
    )
    for param in reference_model.parameters():
        param.requires_grad_(False)
    try:
        from transformers import GPT2TokenizerFast
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Official RL requires transformers. Run `uv sync --extra official`.") from exc
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    prompts = load_prompts(args.prompts_path)
    seq_len = min(args.seq_len, int(model.config.model.length))
    mask_id = int(graph.dim - 1)
    reward_config = {"kind": args.reward_kind, "min_chars": 80}
    baseline = 0.0

    json_log(
        {
            "event": "official_rl_start",
            "device": str(device),
            "model_path": args.model_path,
            "reference_model_path": args.reference_model_path,
            "base_model_path": base_model_path,
            "loaded_step": loaded_step,
            "seq_len": seq_len,
            "prompts": len(prompts),
        }
    )

    for update in tqdm(range(1, args.updates + 1)):
        optimizer.zero_grad(set_to_none=True)
        policy_terms: list[torch.Tensor] = []
        kl_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        rewards: list[float] = []
        samples: list[str] = []
        for _ in range(args.batch_size):
            prompt = random.choice(prompts)
            trace = official_sample_trace(
                model,
                reference_model,
                noise,
                tokenizer,
                prompt,
                seq_len=seq_len,
                mask_id=mask_id,
                max_new_tokens=args.max_new_tokens,
                steps=args.sample_steps,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
            )
            text = tokenizer.decode(trace.response_ids.tolist(), skip_special_tokens=True)
            reward = compute_reward(text, reward_config)
            rewards.append(reward)
            samples.append(text)
            advantage = reward - baseline
            policy_terms.append(-torch.tensor(advantage, device=device) * trace.logprob_sum)
            kl_terms.append(trace.logprob_sum - trace.ref_logprob_sum)
            entropy_terms.append(trace.entropy_sum)
        mean_reward = sum(rewards) / max(len(rewards), 1)
        baseline = 0.9 * baseline + 0.1 * mean_reward
        policy_loss = torch.stack(policy_terms).mean()
        kl_term = torch.stack(kl_terms).mean()
        entropy_term = torch.stack(entropy_terms).mean()
        loss = policy_loss + args.kl_coef * kl_term - args.entropy_coef * entropy_term
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % args.log_every == 0:
            json_log(
                {
                    "event": "official_rl_update",
                    "update": update,
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "kl_term": float(kl_term.detach().cpu()),
                    "entropy": float(entropy_term.detach().cpu()),
                    "reward": mean_reward,
                    "baseline": baseline,
                    "sample": samples[0][:240],
                }
            )
        if update % args.save_every == 0:
            save_official_checkpoint(
                out_dir / f"checkpoint_rl_{update}.pt",
                model=model,
                base_model_path=base_model_path,
                step=update,
                optimizer=optimizer,
                extra={"stage": "rl", "source_model_path": args.model_path},
            )

    save_official_checkpoint(
        out_dir / "checkpoint_last.pt",
        model=model,
        base_model_path=base_model_path,
        step=args.updates,
        optimizer=optimizer,
        extra={"stage": "rl", "source_model_path": args.model_path},
    )
    json_log({"event": "official_rl_done", "checkpoint": str(out_dir / "checkpoint_last.pt")})


if __name__ == "__main__":
    main()
