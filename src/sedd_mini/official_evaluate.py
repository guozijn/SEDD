from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import TokenDataset, collate_batch
from .official_backend import load_official_components
from .official_finetune import evaluate
from .official_posttrain_rl import load_gpt2_tokenizer, official_sample_trace
from .reward import compute_reward
from .utils import get_device, json_log, set_seed


def load_prompts(path: str | Path, limit: int) -> list[str]:
    path = Path(path)
    if not path.exists():
        return []
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return prompts[:limit]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate official SEDD checkpoints.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--official-repo", default="external/Score-Entropy-Discrete-Diffusion")
    parser.add_argument("--data", default="data/processed/official_s1k_valid.pt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--prompts-path", default="")
    parser.add_argument("--reward-kind", default="heuristic_helpfulness")
    parser.add_argument("--reward-samples", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=41)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    model, graph, noise, base_model_path, step = load_official_components(
        args.model_path,
        repo_path=args.official_repo,
        device=device,
    )
    dataset = TokenDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    val_loss = evaluate(
        model,
        graph,
        noise,
        loader,
        device=device,
        max_batches=args.max_batches,
    )
    payload = {
        "event": "official_eval_done",
        "model_path": args.model_path,
        "base_model_path": base_model_path,
        "step": step,
        "data": args.data,
        "score_entropy": val_loss,
        "batches": min(args.max_batches, len(loader)),
    }
    if args.prompts_path and args.reward_samples > 0:
        tokenizer = load_gpt2_tokenizer()
        prompts = load_prompts(args.prompts_path, args.reward_samples)
        rewards: list[float] = []
        texts: list[str] = []
        mask_id = int(graph.dim - 1)
        seq_len = min(args.seq_len, int(model.config.model.length))
        with torch.no_grad():
            for prompt in prompts:
                trace = official_sample_trace(
                    model,
                    model,
                    noise,
                    tokenizer,
                    prompt,
                    seq_len=seq_len,
                    mask_id=mask_id,
                    max_new_tokens=args.max_new_tokens,
                    steps=args.sample_steps,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.95,
                    device=device,
                )
                text = tokenizer.decode(trace.response_ids.tolist(), skip_special_tokens=True)
                rewards.append(compute_reward(text, {"kind": args.reward_kind, "min_chars": 80}))
                texts.append(text[:180])
        payload["reward_kind"] = args.reward_kind
        payload["reward_samples"] = len(rewards)
        payload["mean_reward"] = sum(rewards) / max(len(rewards), 1)
        payload["sample_outputs"] = texts[:3]
    json_log(payload)


if __name__ == "__main__":
    main()
