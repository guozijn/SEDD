from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .mcqa_data import MCQARecord, exact_choice_reward, extract_choice, load_mcqa_records
from .official_backend import load_official_components, save_official_checkpoint
from .sampling import top_k_top_p_filter
from .utils import get_device, json_log, set_seed


def load_gpt2_tokenizer() -> Any:
    try:
        from transformers import GPT2TokenizerFast
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("DCoLT SEDD training requires transformers.") from exc

    try:
        return GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    except Exception:  # noqa: BLE001
        return GPT2TokenizerFast.from_pretrained("gpt2")


@dataclass
class OfficialTrace:
    ids: torch.Tensor
    response_ids: torch.Tensor
    logprob_sum: torch.Tensor
    ref_logprob_sum: torch.Tensor
    entropy_sum: torch.Tensor
    action_count: int


@dataclass
class DCoLTRollout:
    trace: OfficialTrace
    reward: float
    prediction: str
    text: str
    gold: str


def group_normalized_advantages(rewards: list[float], *, eps: float = 1.0e-6) -> list[float]:
    if not rewards:
        return []
    values = torch.tensor(rewards, dtype=torch.float32)
    if values.numel() < 2:
        return [0.0]
    centered = values - values.mean()
    std = values.std(unbiased=False)
    if float(std) < eps:
        return [0.0 for _ in rewards]
    return (centered / (std + eps)).tolist()


def choose_positions(masked: torch.Tensor, step: int, steps: int) -> torch.Tensor:
    positions = masked.nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return positions
    remaining_steps = max(1, steps - step)
    count = max(1, int(torch.ceil(torch.tensor(positions.numel() / remaining_steps)).item()))
    order = torch.randperm(positions.numel(), device=positions.device)
    return positions[order[:count]]


def filtered_log_probs(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
    scaled = logits.float()
    filtered = top_k_top_p_filter(scaled, top_k=top_k, top_p=top_p)
    log_probs = F.log_softmax(filtered, dim=-1)
    probs = log_probs.exp()
    bad_rows = (~torch.isfinite(probs).all(dim=-1)) | (probs.sum(dim=-1) <= 0)
    if bad_rows.any():
        log_probs = log_probs.clone()
        log_probs[bad_rows] = F.log_softmax(scaled[bad_rows], dim=-1)
    return log_probs


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
    action_count = 0

    for step in range(max(1, steps)):
        masked = (ids[0] == mask_id) & response_mask
        fill_positions = choose_positions(masked, step, max(1, steps))
        if fill_positions.numel() == 0:
            break

        t = torch.full((1,), 1.0 - (step / max(steps, 1)) * 0.999, device=device)
        sigma = noise(t)[0]
        logits = model(ids.clone(), sigma)[0, fill_positions, :mask_id]
        logits = logits / max(temperature, 1.0e-5)
        log_probs = filtered_log_probs(logits, top_k=top_k, top_p=top_p)
        probs = log_probs.exp()
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        selected_log_probs = log_probs.gather(1, sampled[:, None]).squeeze(-1)
        safe_log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))
        entropy = -(probs * safe_log_probs).sum(dim=-1)

        with torch.no_grad():
            ref_logits = reference_model(ids.clone(), sigma)[0, fill_positions, :mask_id]
            ref_logits = ref_logits / max(temperature, 1.0e-5)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            selected_ref_log_probs = ref_log_probs.gather(1, sampled[:, None]).squeeze(-1)

        logprob_sum = logprob_sum + selected_log_probs.sum()
        ref_logprob_sum = ref_logprob_sum + selected_ref_log_probs.sum()
        entropy_sum = entropy_sum + entropy.sum()
        action_count += int(sampled.numel())
        ids[0, fill_positions] = sampled

    ids[ids == mask_id] = eos
    return OfficialTrace(
        ids=ids.detach().clone(),
        response_ids=ids[0, response_slice].detach().clone(),
        logprob_sum=logprob_sum,
        ref_logprob_sum=ref_logprob_sum,
        entropy_sum=entropy_sum,
        action_count=action_count,
    )


def rollout_record(
    *,
    model: torch.nn.Module,
    reference_model: torch.nn.Module,
    noise: Any,
    tokenizer: Any,
    record: MCQARecord,
    seq_len: int,
    mask_id: int,
    max_new_tokens: int,
    steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> DCoLTRollout:
    trace = official_sample_trace(
        model,
        reference_model,
        noise,
        tokenizer,
        record.prompt,
        seq_len=seq_len,
        mask_id=mask_id,
        max_new_tokens=max_new_tokens,
        steps=steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
    )
    text = tokenizer.decode(trace.response_ids.tolist(), skip_special_tokens=True)
    prediction = extract_choice(text, record.labels)
    reward = exact_choice_reward(text, record.answer, record.labels)
    return DCoLTRollout(
        trace=trace,
        reward=reward,
        prediction=prediction,
        text=text,
        gold=record.answer,
    )


def dcolt_loss(
    rollouts: list[DCoLTRollout],
    advantages: list[float],
    *,
    clip_eps: float,
    beta: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: list[torch.Tensor] = []
    policy_losses: list[torch.Tensor] = []
    kl_terms: list[torch.Tensor] = []
    entropy_terms: list[torch.Tensor] = []

    for rollout, advantage in zip(rollouts, advantages, strict=True):
        trace = rollout.trace
        action_count = max(1, trace.action_count)
        logprob = trace.logprob_sum / action_count
        old_logprob = logprob.detach()
        ratio = torch.exp((logprob - old_logprob).clamp(min=-20.0, max=20.0))
        advantage_tensor = torch.tensor(float(advantage), device=logprob.device)
        unclipped = ratio * advantage_tensor
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage_tensor
        policy_loss = -torch.minimum(unclipped, clipped)
        kl_term = (trace.logprob_sum - trace.ref_logprob_sum) / action_count
        entropy_term = trace.entropy_sum / action_count
        loss = policy_loss + beta * kl_term - entropy_coef * entropy_term
        losses.append(loss)
        policy_losses.append(policy_loss.detach())
        kl_terms.append(kl_term.detach())
        entropy_terms.append(entropy_term.detach())

    total = torch.stack(losses).mean()
    return total, {
        "policy_loss": float(torch.stack(policy_losses).mean().cpu()) if policy_losses else 0.0,
        "kl_term": float(torch.stack(kl_terms).mean().cpu()) if kl_terms else 0.0,
        "entropy": float(torch.stack(entropy_terms).mean().cpu()) if entropy_terms else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DCoLT post-train an official SEDD checkpoint.")
    parser.add_argument("--model-path", default="runs/arc_models/arc_sft/checkpoint_last.pt")
    parser.add_argument("--reference-model-path", default="runs/arc_models/base/checkpoint_base.pt")
    parser.add_argument("--official-repo", default="external/Score-Entropy-Discrete-Diffusion")
    parser.add_argument("--records-path", default="data/processed/arc_challenge_rl_train.jsonl")
    parser.add_argument("--out-dir", default="runs/arc_models/arc_dcolt_rl")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1, help="Prompt groups per optimizer update.")
    parser.add_argument("--num-generations", type=int, default=4, help="Rollouts per prompt.")
    parser.add_argument("--repeat-times", type=int, default=1, help="Extra rollout repeats per prompt.")
    parser.add_argument("--lr", type=float, default=5.0e-6)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--sample-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=25)
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

    records = load_mcqa_records(args.records_path)
    if not records:
        raise ValueError(f"no MCQA records found at {args.records_path}")

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
    model.train()
    reference_model.eval()

    tokenizer = load_gpt2_tokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    seq_len = min(args.seq_len, int(model.config.model.length))
    mask_id = int(graph.dim - 1)
    rollouts_per_prompt = max(1, args.num_generations) * max(1, args.repeat_times)

    json_log(
        {
            "event": "dcolt_start",
            "device": str(device),
            "model_path": args.model_path,
            "reference_model_path": args.reference_model_path,
            "base_model_path": base_model_path,
            "loaded_step": loaded_step,
            "records_path": args.records_path,
            "records": len(records),
            "seq_len": seq_len,
            "max_new_tokens": args.max_new_tokens,
            "sample_steps": args.sample_steps,
            "batch_size": args.batch_size,
            "num_generations": args.num_generations,
            "repeat_times": args.repeat_times,
            "rollouts_per_prompt": rollouts_per_prompt,
            "clip_eps": args.clip_eps,
            "beta": args.beta,
        }
    )

    for update in tqdm(range(1, args.updates + 1), desc="official sedd dcolt"):
        optimizer.zero_grad(set_to_none=True)
        update_rollouts: list[DCoLTRollout] = []
        update_advantages: list[float] = []

        for _ in range(max(1, args.batch_size)):
            record = random.choice(records)
            group = [
                rollout_record(
                    model=model,
                    reference_model=reference_model,
                    noise=noise,
                    tokenizer=tokenizer,
                    record=record,
                    seq_len=seq_len,
                    mask_id=mask_id,
                    max_new_tokens=args.max_new_tokens,
                    steps=args.sample_steps,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=device,
                )
                for _ in range(rollouts_per_prompt)
            ]
            update_rollouts.extend(group)
            update_advantages.extend(group_normalized_advantages([rollout.reward for rollout in group]))

        valid_samples = sum(1 for advantage in update_advantages if abs(advantage) > 1.0e-8)
        if valid_samples == 0:
            json_log(
                {
                    "event": "dcolt_skip",
                    "update": update,
                    "reason": "zero_group_variance",
                    "reward": sum(rollout.reward for rollout in update_rollouts)
                    / max(len(update_rollouts), 1),
                }
            )
            continue

        loss, metrics = dcolt_loss(
            update_rollouts,
            update_advantages,
            clip_eps=args.clip_eps,
            beta=args.beta,
            entropy_coef=args.entropy_coef,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        rewards = [rollout.reward for rollout in update_rollouts]
        if update % args.log_every == 0:
            sample = update_rollouts[0]
            payload = {
                "event": "dcolt_update",
                "update": update,
                "loss": float(loss.detach().cpu()),
                "reward": sum(rewards) / max(len(rewards), 1),
                "reward_min": min(rewards),
                "reward_max": max(rewards),
                "valid_samples": valid_samples,
                "gold": sample.gold,
                "pred": sample.prediction or "?",
                "sample": sample.text[:180],
            }
            payload.update(metrics)
            json_log(payload)

        if update % args.save_every == 0:
            save_official_checkpoint(
                out_dir / f"checkpoint_dcolt_{update}.pt",
                model=model,
                base_model_path=base_model_path,
                step=update,
                optimizer=optimizer,
                extra={
                    "stage": "dcolt_rl",
                    "source_model_path": args.model_path,
                    "reference_model_path": args.reference_model_path,
                    "records_path": args.records_path,
                    "num_generations": args.num_generations,
                    "repeat_times": args.repeat_times,
                    "clip_eps": args.clip_eps,
                    "beta": args.beta,
                },
            )

    save_official_checkpoint(
        out_dir / "checkpoint_last.pt",
        model=model,
        base_model_path=base_model_path,
        step=args.updates,
        optimizer=optimizer,
        extra={
            "stage": "dcolt_rl",
            "source_model_path": args.model_path,
            "reference_model_path": args.reference_model_path,
            "records_path": args.records_path,
            "num_generations": args.num_generations,
            "repeat_times": args.repeat_times,
            "clip_eps": args.clip_eps,
            "beta": args.beta,
        },
    )
    json_log({"event": "dcolt_done", "checkpoint": str(out_dir / "checkpoint_last.pt")})


if __name__ == "__main__":
    main()
