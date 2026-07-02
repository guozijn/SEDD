from __future__ import annotations

from dataclasses import dataclass

import torch

from .checkpoint import load_checkpoint
from .diffusion import loglinear_noise
from .official_backend import OfficialSEDDBackend
from .sampling import infill_text, prompt_to_ids, sample_response, top_k_top_p_filter
from .tokenizer import ByteTokenizer
from .utils import get_device, set_seed


@dataclass
class GenerationParams:
    max_new_tokens: int = 96
    steps: int = 32
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.95


class MiniBackend:
    name = "mini"

    def __init__(self, checkpoint: str, *, device_name: str = "auto", seed: int = 29) -> None:
        set_seed(seed)
        self.device = get_device(device_name)
        self.model, self.config, self.payload = load_checkpoint(
            checkpoint, device=self.device, use_ema=True
        )
        self.tokenizer = ByteTokenizer()
        self.checkpoint = checkpoint

    def health(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "device": str(self.device),
            "checkpoint": self.checkpoint,
            "step": int(self.payload.get("step", 0)),
            "seq_len": int(self.config["model"]["seq_len"]),
        }

    def generate(self, prompt: str, params: GenerationParams) -> str:
        with torch.no_grad():
            return str(
                sample_response(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_new_tokens=params.max_new_tokens,
                    steps=params.steps,
                    temperature=params.temperature,
                    top_k=params.top_k,
                    top_p=params.top_p,
                    seq_len=int(self.config["model"]["seq_len"]),
                    device=self.device,
                )
            )

    def infill(self, text: str, params: GenerationParams) -> str:
        with torch.no_grad():
            return infill_text(
                self.model,
                self.tokenizer,
                text,
                max_fill_tokens=params.max_new_tokens,
                steps=params.steps,
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
                seq_len=int(self.config["model"]["seq_len"]),
                device=self.device,
            )

    def visualize_generate(
        self, prompt: str, params: GenerationParams, *, batch_size: int = 4
    ) -> dict[str, object]:
        seq_len = int(self.config["model"]["seq_len"])
        max_prompt = max(1, seq_len - params.max_new_tokens)
        prompt_ids = prompt_to_ids(prompt, self.tokenizer, max_prompt)
        gen_len = min(params.max_new_tokens, seq_len - len(prompt_ids))
        ids = torch.full((batch_size, seq_len), self.tokenizer.pad_id, dtype=torch.long, device=self.device)
        prefix = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        ids[:, : len(prompt_ids)] = prefix
        response_slice = slice(len(prompt_ids), len(prompt_ids) + gen_len)
        ids[:, response_slice] = self.tokenizer.mask_id
        response_mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        response_mask[response_slice] = True

        def token_label(token_id: int) -> str:
            if token_id == self.tokenizer.mask_id:
                return "[MASK]"
            if token_id == self.tokenizer.pad_id:
                return ""
            if token_id == self.tokenizer.eos_id:
                return "<eos>"
            text = self.tokenizer.decode([token_id], stop_at_eos=False)
            return text if text.strip() else repr(text)[1:-1]

        def snapshot(step: int) -> dict[str, object]:
            samples = []
            for row in ids[:, response_slice].detach().cpu().tolist():
                samples.append(
                    {
                        "text": self.tokenizer.decode(row),
                        "tokens": [token_label(token) for token in row],
                        "masked": sum(1 for token in row if token == self.tokenizer.mask_id),
                    }
                )
            return {"step": step, "samples": samples}

        trace = [snapshot(0)]
        self.model.eval()
        with torch.no_grad():
            for step in range(params.steps):
                masked_any = False
                t = torch.full(
                    (batch_size,),
                    1.0 - (step / max(params.steps, 1)) * 0.999,
                    device=self.device,
                )
                sigma = loglinear_noise(t)[0]
                logits_all = self.model(ids.clone(), sigma)[..., : self.tokenizer.mask_id]
                remaining_steps = max(1, params.steps - step)
                for batch_idx in range(batch_size):
                    masked = (ids[batch_idx] == self.tokenizer.mask_id) & response_mask
                    positions = masked.nonzero(as_tuple=False).flatten()
                    if positions.numel() == 0:
                        continue
                    masked_any = True
                    count = max(1, int(torch.ceil(torch.tensor(positions.numel() / remaining_steps)).item()))
                    order = torch.randperm(positions.numel(), device=self.device)
                    fill_positions = positions[order[:count]]
                    logits = logits_all[batch_idx, fill_positions] / max(params.temperature, 1.0e-5)
                    logits = top_k_top_p_filter(logits, top_k=params.top_k, top_p=params.top_p)
                    probs = torch.softmax(logits, dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    ids[batch_idx, fill_positions] = sampled
                trace.append(snapshot(step + 1))
                if not masked_any:
                    break

        return {
            "backend": self.name,
            "prompt": prompt,
            "batch_size": batch_size,
            "response_tokens": gen_len,
            "steps": trace,
        }


def create_backend(
    backend: str = "official",
    *,
    checkpoint: str = "",
    model_path: str = "louaaron/sedd-small",
    official_repo: str = "external/Score-Entropy-Discrete-Diffusion",
    device_name: str = "auto",
    seed: int = 29,
):
    if backend == "mini":
        if not checkpoint:
            raise ValueError("--checkpoint is required for --backend mini")
        return MiniBackend(checkpoint, device_name=device_name, seed=seed)
    if backend == "official":
        return OfficialSEDDBackend(
            model_path=model_path,
            repo_path=official_repo,
            device_name=device_name,
            seed=seed,
        )
    raise ValueError(f"unknown backend: {backend}")
