from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .utils import get_device, set_seed


class OfficialBackendError(RuntimeError):
    pass


@dataclass
class OfficialModules:
    load_model: Any
    sampling: Any
    tokenizer_cls: Any
    sedd_cls: Any
    graph_lib: Any
    noise_lib: Any
    omegaconf: Any


def _install_flash_attn_fallback() -> None:
    """Install a minimal flash-attn shim for the function used by upstream SEDD.

    The official SEDD code imports `flash_attn_varlen_qkvpacked_func` at module
    import time. On machines where flash-attn is unavailable or difficult to
    compile, this fallback maps that call to PyTorch SDPA. It is slower but
    correct for validation and small batch demos.
    """

    try:
        if importlib.util.find_spec("flash_attn.flash_attn_interface") is not None:
            return
    except (ImportError, ModuleNotFoundError, ValueError):
        pass

    flash_attn_module = types.ModuleType("flash_attn")
    flash_attn_module.__path__ = []  # type: ignore[attr-defined]
    interface_module = types.ModuleType("flash_attn.flash_attn_interface")
    layers_module = types.ModuleType("flash_attn.layers")
    layers_module.__path__ = []  # type: ignore[attr-defined]
    rotary_module = types.ModuleType("flash_attn.layers.rotary")

    def flash_attn_varlen_qkvpacked_func(
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        dropout_p: float,
        causal: bool = False,
    ) -> torch.Tensor:
        del max_seqlen
        outputs: list[torch.Tensor] = []
        batch = int(cu_seqlens.numel() - 1)
        for idx in range(batch):
            start = int(cu_seqlens[idx].item())
            end = int(cu_seqlens[idx + 1].item())
            chunk = qkv[start:end]
            q, k, v = chunk.unbind(dim=1)
            q = q.permute(1, 0, 2).unsqueeze(0)
            k = k.permute(1, 0, 2).unsqueeze(0)
            v = v.permute(1, 0, 2).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_p if q.requires_grad else 0.0,
                is_causal=causal,
            )
            outputs.append(out.squeeze(0).permute(1, 0, 2))
        return torch.cat(outputs, dim=0)

    interface_module.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_emb_qkv_(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cos_full = torch.cat([cos, cos], dim=-1).to(dtype=qkv.dtype, device=qkv.device)
        sin_full = torch.cat([sin, sin], dim=-1).to(dtype=qkv.dtype, device=qkv.device)
        cos_full = cos_full[None, :, None, :]
        sin_full = sin_full[None, :, None, :]
        qkv = qkv.clone()
        qkv[:, :, 0] = (qkv[:, :, 0] * cos_full) + (rotate_half(qkv[:, :, 0]) * sin_full)
        qkv[:, :, 1] = (qkv[:, :, 1] * cos_full) + (rotate_half(qkv[:, :, 1]) * sin_full)
        return qkv

    rotary_module.apply_rotary_emb_qkv_ = apply_rotary_emb_qkv_
    layers_module.rotary = rotary_module
    flash_attn_module.flash_attn_interface = interface_module
    flash_attn_module.layers = layers_module
    sys.modules.setdefault("flash_attn", flash_attn_module)
    sys.modules.setdefault("flash_attn.flash_attn_interface", interface_module)
    sys.modules.setdefault("flash_attn.layers", layers_module)
    sys.modules.setdefault("flash_attn.layers.rotary", rotary_module)


def _import_official_modules(repo_path: str | Path) -> OfficialModules:
    repo_path = Path(repo_path).expanduser().resolve()
    if not repo_path.exists():
        raise OfficialBackendError(
            f"Official SEDD repo not found at {repo_path}. "
            "Run scripts/setup_official_backend.sh first or pass --official-repo."
        )
    sys.path.insert(0, str(repo_path))
    _install_flash_attn_fallback()
    try:
        load_model = importlib.import_module("load_model")
        sampling = importlib.import_module("sampling")
        transformers = importlib.import_module("transformers")
        model_module = importlib.import_module("model")
        graph_lib = importlib.import_module("graph_lib")
        noise_lib = importlib.import_module("noise_lib")
        omegaconf = importlib.import_module("omegaconf")
    except Exception as exc:  # noqa: BLE001
        raise OfficialBackendError(
            "Could not import official SEDD backend. Install optional dependencies with "
            "`uv sync --extra official`, and on CUDA install `flash-attn` compatible with "
            "your PyTorch/CUDA build. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return OfficialModules(
        load_model=load_model.load_model,
        sampling=sampling,
        tokenizer_cls=transformers.GPT2TokenizerFast,
        sedd_cls=model_module.SEDD,
        graph_lib=graph_lib,
        noise_lib=noise_lib,
        omegaconf=omegaconf.OmegaConf,
    )


class OfficialSEDDBackend:
    """Lazy wrapper around louaaron/Score-Entropy-Discrete-Diffusion.

    This backend intentionally depends on the official repository and its CUDA
    stack. It is kept separate from the mini backend so local CPU/MPS smoke tests
    remain lightweight and reproducible.
    """

    name = "official"

    def __init__(
        self,
        *,
        model_path: str = "louaaron/sedd-small",
        repo_path: str | Path = "external/Score-Entropy-Discrete-Diffusion",
        device_name: str = "auto",
        seed: int = 29,
    ) -> None:
        set_seed(seed)
        self.device = get_device(device_name)
        if self.device.type != "cuda":
            print(
                "warning: official SEDD backend is much slower without CUDA; "
                "use --device cuda on the remote GPU for practical sampling.",
                file=sys.stderr,
            )
        self.model_path = model_path
        self.repo_path = str(repo_path)
        modules = _import_official_modules(repo_path)
        self._sampling = modules.sampling
        try:
            self.tokenizer = modules.tokenizer_cls.from_pretrained("gpt2", local_files_only=True)
        except Exception:  # noqa: BLE001
            self.tokenizer = modules.tokenizer_cls.from_pretrained("gpt2")
        self.model, self.graph, self.noise, self.base_model_path, self.step = load_official_components(
            model_path,
            repo_path=repo_path,
            device=self.device,
            modules=modules,
        )
        self.model.eval()
        self.seq_len = int(self.model.config.model.length)

    def health(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "device": str(self.device),
            "model_path": self.model_path,
            "base_model_path": self.base_model_path,
            "official_repo": self.repo_path,
            "seq_len": self.seq_len,
            "step": self.step,
        }

    def _sample_with_constraints(
        self,
        *,
        prefix: str,
        suffix: str = "",
        batch_size: int = 1,
        steps: int = 128,
        predictor: str = "analytic",
    ) -> torch.Tensor:
        prefix_ids = self.tokenizer(prefix).input_ids
        suffix_ids = self.tokenizer(suffix).input_ids if suffix else []
        if len(prefix_ids) + len(suffix_ids) >= self.seq_len:
            keep_prefix = max(0, self.seq_len - len(suffix_ids) - 1)
            prefix_ids = prefix_ids[-keep_prefix:]

        input_ids = prefix_ids + suffix_ids
        input_locs = list(range(len(prefix_ids)))
        if suffix_ids:
            input_locs += list(range(self.seq_len - len(suffix_ids), self.seq_len))

        if input_ids:
            fixed = torch.tensor(input_ids, device=self.device)[None].repeat(batch_size, 1)
        else:
            fixed = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)

        def project(x: torch.Tensor) -> torch.Tensor:
            if input_locs:
                x[:, input_locs] = fixed
            return x

        sampler = self._sampling.get_pc_sampler(
            self.graph,
            self.noise,
            (batch_size, self.seq_len),
            predictor,
            steps,
            device=self.device,
            proj_fun=project,
        )
        return project(sampler(self.model))

    @torch.no_grad()
    def generate(self, prompt: str, params: Any) -> str:
        prefix_ids = self.tokenizer(prompt).input_ids
        if len(prefix_ids) >= self.seq_len:
            prefix_len = self.seq_len - 1
        else:
            prefix_len = len(prefix_ids)
        samples = self._sample_with_constraints(
            prefix=prompt,
            steps=params.steps,
            predictor="analytic",
        )
        end = min(self.seq_len, prefix_len + int(params.max_new_tokens))
        return self.tokenizer.decode(samples[0, prefix_len:end], skip_special_tokens=True).strip()

    @torch.no_grad()
    def infill(self, text: str, params: Any) -> str:
        marker = "[MASK]"
        if marker not in text:
            return self.generate(text, params)
        prefix, suffix = text.split(marker, 1)
        samples = self._sample_with_constraints(
            prefix=prefix,
            suffix=suffix,
            steps=params.steps,
            predictor="analytic",
        )
        return self.tokenizer.batch_decode(samples)[0].strip()

    @torch.no_grad()
    def visualize_generate(
        self, prompt: str, params: Any, *, batch_size: int = 4
    ) -> dict[str, object]:
        from .sampling import top_k_top_p_filter

        eos = int(self.tokenizer.eos_token_id)
        mask_id = int(self.graph.dim - 1)
        prefix_ids = self.tokenizer(prompt).input_ids
        max_prompt = max(1, self.seq_len - int(params.max_new_tokens))
        prefix_ids = prefix_ids[-max_prompt:]
        gen_len = min(int(params.max_new_tokens), self.seq_len - len(prefix_ids))
        ids = torch.full((batch_size, self.seq_len), eos, dtype=torch.long, device=self.device)
        ids[:, : len(prefix_ids)] = torch.tensor(prefix_ids, dtype=torch.long, device=self.device)
        response_slice = slice(len(prefix_ids), len(prefix_ids) + gen_len)
        ids[:, response_slice] = mask_id
        response_mask = torch.zeros(self.seq_len, dtype=torch.bool, device=self.device)
        response_mask[response_slice] = True

        def token_label(token_id: int) -> str:
            if token_id == mask_id:
                return "[MASK]"
            if token_id == eos:
                return "<eos>"
            text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            return text if text.strip() else repr(text)[1:-1]

        def snapshot(step: int) -> dict[str, object]:
            samples = []
            for row in ids[:, response_slice].detach().cpu().tolist():
                samples.append(
                    {
                        "text": self.tokenizer.decode(row, skip_special_tokens=True),
                        "tokens": [token_label(token) for token in row],
                        "masked": sum(1 for token in row if token == mask_id),
                    }
                )
            return {"step": step, "samples": samples}

        trace = [snapshot(0)]
        self.model.eval()
        for step in range(int(params.steps)):
            masked_any = False
            t = torch.full(
                (batch_size,),
                1.0 - (step / max(int(params.steps), 1)) * 0.999,
                device=self.device,
            )
            sigma = self.noise(t)[0]
            logits_all = self.model(ids.clone(), sigma)[..., :mask_id]
            remaining_steps = max(1, int(params.steps) - step)
            for batch_idx in range(batch_size):
                masked = (ids[batch_idx] == mask_id) & response_mask
                positions = masked.nonzero(as_tuple=False).flatten()
                if positions.numel() == 0:
                    continue
                masked_any = True
                count = max(1, int(torch.ceil(torch.tensor(positions.numel() / remaining_steps)).item()))
                order = torch.randperm(positions.numel(), device=self.device)
                fill_positions = positions[order[:count]]
                logits = logits_all[batch_idx, fill_positions] / max(float(params.temperature), 1.0e-5)
                logits = top_k_top_p_filter(logits, top_k=int(params.top_k), top_p=float(params.top_p))
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


def check_main() -> None:
    parser = argparse.ArgumentParser(description="Check official SEDD backend availability.")
    parser.add_argument("--model-path", default="louaaron/sedd-small")
    parser.add_argument("--official-repo", default="external/Score-Entropy-Discrete-Diffusion")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    backend = OfficialSEDDBackend(
        model_path=args.model_path,
        repo_path=args.official_repo,
        device_name=args.device,
    )
    print(backend.health())


def load_official_components(
    model_path: str,
    *,
    repo_path: str | Path,
    device: torch.device,
    modules: OfficialModules | None = None,
):
    modules = modules or _import_official_modules(repo_path)
    path = Path(model_path).expanduser()
    if path.is_file():
        payload = torch.load(path, map_location="cpu")
        base_model_path = payload.get("base_model_path", "louaaron/sedd-small")
        config = _load_official_model_config(base_model_path, modules)
        graph = modules.graph_lib.get_graph(config, device)
        noise = modules.noise_lib.get_noise(config).to(device)
        model = modules.sedd_cls(config).to(device)
        model.load_state_dict(payload["model"], strict=True)
        return model, graph, noise, base_model_path, int(payload.get("step", 0))
    model, graph, noise = modules.load_model(model_path, device)
    return model, graph, noise, model_path, 0


def _load_official_model_config(model_path: str, modules: OfficialModules):
    path = Path(str(model_path)).expanduser()
    candidates = []
    if path.is_dir():
        candidates.append(path / "config.json")
    if path.is_file() and path.name == "config.json":
        candidates.append(path)

    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as f:
                return modules.omegaconf.create(json.load(f))

    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
        config_path = huggingface_hub.hf_hub_download(
            repo_id=str(model_path),
            filename="config.json",
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise OfficialBackendError(
            f"Could not find cached official SEDD config for {model_path!r}. "
            "Run `scripts/setup_official_backend.sh` with network/proxy once, "
            "or pass a local config directory."
        ) from exc

    with Path(config_path).open("r", encoding="utf-8") as f:
        return modules.omegaconf.create(json.load(f))


def save_official_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    base_model_path: str,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "backend": "official",
        "base_model_path": base_model_path,
        "model": model.state_dict(),
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
