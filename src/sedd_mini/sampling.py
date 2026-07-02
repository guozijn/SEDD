from __future__ import annotations

from dataclasses import dataclass

import torch

from .diffusion import loglinear_noise
from .tokenizer import ByteTokenizer


@dataclass
class SampleTrace:
    token_ids: torch.Tensor
    response_ids: torch.Tensor
    logprob_sum: torch.Tensor
    ref_logprob_sum: torch.Tensor | None
    entropy_sum: torch.Tensor
    steps: int


def top_k_top_p_filter(logits: torch.Tensor, *, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    filtered = logits.clone()
    if top_k and top_k > 0 and top_k < filtered.shape[-1]:
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_filtered = sorted_logits.masked_fill(remove, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_filtered)
    return filtered


def _positions_to_fill(masked: torch.Tensor, step: int, steps: int) -> torch.Tensor:
    remaining = masked.nonzero(as_tuple=False).flatten()
    if remaining.numel() == 0:
        return remaining
    remaining_steps = max(1, steps - step)
    count = max(1, int(torch.ceil(torch.tensor(remaining.numel() / remaining_steps)).item()))
    return remaining[:count]


def prompt_to_ids(prompt: str, tokenizer: ByteTokenizer, max_prompt_tokens: int) -> list[int]:
    ids = tokenizer.encode(f"User: {prompt}\nAssistant: ", add_bos=True, add_eos=False)
    return ids[-max_prompt_tokens:]


def sample_response(
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seq_len: int,
    device: torch.device,
    reference_model: torch.nn.Module | None = None,
    generator: torch.Generator | None = None,
    return_trace: bool = False,
) -> str | SampleTrace:
    model.eval()
    if reference_model is not None:
        reference_model.eval()

    max_prompt = max(1, seq_len - max_new_tokens)
    prompt_ids = prompt_to_ids(prompt, tokenizer, max_prompt)
    gen_len = min(max_new_tokens, seq_len - len(prompt_ids))
    if gen_len <= 0:
        gen_len = 1
        prompt_ids = prompt_ids[-(seq_len - 1) :]
    ids = torch.full((1, seq_len), tokenizer.pad_id, dtype=torch.long, device=device)
    prefix = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    ids[0, : len(prompt_ids)] = prefix
    response_slice = slice(len(prompt_ids), len(prompt_ids) + gen_len)
    ids[0, response_slice] = tokenizer.mask_id

    logprob_sum = torch.zeros((), device=device)
    ref_logprob_sum = torch.zeros((), device=device) if reference_model is not None else None
    entropy_sum = torch.zeros((), device=device)
    response_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    response_mask[response_slice] = True

    for step in range(steps):
        masked = (ids[0] == tokenizer.mask_id) & response_mask
        fill_positions = _positions_to_fill(masked, step, steps)
        if fill_positions.numel() == 0:
            break
        t = torch.full((1,), 1.0 - (step / max(steps, 1)) * 0.999, device=device)
        sigma = loglinear_noise(t)[0]
        logits = model(ids.clone(), sigma)[0, fill_positions, : tokenizer.mask_id]
        logits = logits / max(temperature, 1.0e-5)
        logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
        action_logprobs = torch.log(torch.gather(probs, 1, sampled[:, None]).squeeze(-1).clamp_min(1.0e-8))
        entropy = -(probs * torch.log(probs.clamp_min(1.0e-8))).sum(dim=-1)
        logprob_sum = logprob_sum + action_logprobs.sum()
        entropy_sum = entropy_sum + entropy.sum()
        if reference_model is not None:
            with torch.no_grad():
                ref_logits = reference_model(ids.clone(), sigma)[0, fill_positions, : tokenizer.mask_id]
                ref_probs = torch.softmax(ref_logits / max(temperature, 1.0e-5), dim=-1)
                ref_logprobs = torch.log(
                    torch.gather(ref_probs, 1, sampled[:, None]).squeeze(-1).clamp_min(1.0e-8)
                )
            ref_logprob_sum = ref_logprob_sum + ref_logprobs.sum()
        ids[0, fill_positions] = sampled

    ids[ids == tokenizer.mask_id] = tokenizer.eos_id
    response_ids = ids[0, response_slice].detach().clone()
    if return_trace:
        return SampleTrace(
            token_ids=ids.detach().clone(),
            response_ids=response_ids,
            logprob_sum=logprob_sum,
            ref_logprob_sum=ref_logprob_sum,
            entropy_sum=entropy_sum,
            steps=steps,
        )
    return tokenizer.decode(response_ids.tolist())


@torch.no_grad()
def infill_text(
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    text_with_mask: str,
    *,
    mask: str = "[MASK]",
    max_fill_tokens: int = 32,
    steps: int = 16,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    seq_len: int,
    device: torch.device,
) -> str:
    if mask not in text_with_mask:
        return text_with_mask
    before, after = text_with_mask.split(mask, 1)
    prompt = before
    trace = sample_response(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_fill_tokens,
        steps=steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seq_len=seq_len,
        device=device,
        return_trace=True,
    )
    assert isinstance(trace, SampleTrace)
    fill = tokenizer.decode(trace.response_ids.tolist())
    return before + fill + after
