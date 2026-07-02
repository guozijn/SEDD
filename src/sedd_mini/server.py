from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .backend import GenerationParams, create_backend

APP_DIR = Path(__file__).resolve().parents[2] / "app" / "frontend"
DEFAULT_REGISTRY_CANDIDATES = (
    Path("runs/arc_models/registry.json"),
)


class GenerateRequest(BaseModel):
    model_id: str | None = None
    prompt: str
    max_new_tokens: int = 96
    steps: int = 32
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.95


class InfillRequest(BaseModel):
    model_id: str | None = None
    text: str
    max_fill_tokens: int = 48
    steps: int = 24
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.95


class VisualizeRequest(BaseModel):
    model_id: str | None = None
    prompt: str
    batch_size: int = 4
    max_new_tokens: int = 32
    steps: int = 8
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.95


def load_registry(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    models = payload.get("models") or []
    if not models:
        raise ValueError(f"model registry {path} has no models")
    return payload


def discover_model_registry() -> str:
    for candidate in DEFAULT_REGISTRY_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return ""


def create_app(
    checkpoint: str,
    device_name: str = "auto",
    seed: int = 29,
    *,
    backend_name: str = "official",
    model_path: str = "louaaron/sedd-small",
    official_repo: str = "external/Score-Entropy-Discrete-Diffusion",
    model_registry: str = "",
    default_model_id: str = "",
) -> FastAPI:
    model_registry = model_registry or discover_model_registry()
    registry = load_registry(model_registry) if model_registry else None
    backend_cache: dict[str, Any] = {}

    def registry_models() -> list[dict[str, Any]]:
        if not registry:
            return [
                {
                    "id": "default",
                    "label": model_path if backend_name == "official" else checkpoint,
                    "backend": backend_name,
                }
            ]
        return registry["models"]

    def resolve_model_id(model_id: str | None = None) -> str:
        if not registry:
            return "default"
        if model_id:
            return model_id
        if default_model_id:
            return default_model_id
        if registry.get("default_model_id"):
            return str(registry["default_model_id"])
        return str(registry["models"][0]["id"])

    def get_backend(model_id: str | None = None):
        resolved = resolve_model_id(model_id)
        if resolved in backend_cache:
            return backend_cache[resolved]
        if not registry:
            backend = create_backend(
                backend_name,
                checkpoint=checkpoint,
                model_path=model_path,
                official_repo=official_repo,
                device_name=device_name,
                seed=seed,
            )
            backend_cache[resolved] = backend
            return backend
        entry = next((item for item in registry["models"] if item["id"] == resolved), None)
        if entry is None:
            raise ValueError(f"unknown model id: {resolved}")
        backend = create_backend(
            entry.get("backend", "official"),
            checkpoint=entry.get("checkpoint", ""),
            model_path=entry.get("model_path", "louaaron/sedd-small"),
            official_repo=entry.get("official_repo", official_repo),
            device_name=entry.get("device", device_name),
            seed=seed,
        )
        backend_cache[resolved] = backend
        return backend

    app = FastAPI(title="SEDD Demo")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if APP_DIR.exists():
        app.mount("/static", StaticFiles(directory=APP_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(APP_DIR / "index.html")

    @app.get("/health")
    def health():
        payload = get_backend(default_model_id or None).health()
        payload["status"] = "ok"
        payload["default_model_id"] = resolve_model_id(default_model_id or None)
        return payload

    @app.get("/models")
    def models():
        return {
            "default_model_id": resolve_model_id(default_model_id or None),
            "models": registry_models(),
        }

    @app.post("/generate")
    def generate(req: GenerateRequest):
        params = GenerationParams(
            max_new_tokens=req.max_new_tokens,
            steps=req.steps,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        text = get_backend(req.model_id).generate(req.prompt, params)
        return {"text": text}

    @app.post("/infill")
    def infill(req: InfillRequest):
        params = GenerationParams(
            max_new_tokens=req.max_fill_tokens,
            steps=req.steps,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        text = get_backend(req.model_id).infill(req.text, params)
        return {"text": text}

    @app.post("/visualize")
    def visualize(req: VisualizeRequest):
        params = GenerationParams(
            max_new_tokens=req.max_new_tokens,
            steps=req.steps,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        return get_backend(req.model_id).visualize_generate(
            req.prompt,
            params,
            batch_size=max(1, min(req.batch_size, 8)),
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve SEDD demo API/frontend.")
    parser.add_argument("--backend", choices=["mini", "official"], default="official")
    parser.add_argument("--checkpoint", default="", help="Mini backend checkpoint path.")
    parser.add_argument("--model-path", default="louaaron/sedd-small", help="Official HF model path.")
    parser.add_argument("--official-repo", default="external/Score-Entropy-Discrete-Diffusion")
    parser.add_argument("--model-registry", default="", help="JSON registry for multiple demo models.")
    parser.add_argument("--default-model-id", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=29)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = create_app(
        args.checkpoint,
        device_name=args.device,
        seed=args.seed,
        backend_name=args.backend,
        model_path=args.model_path,
        official_repo=args.official_repo,
        model_registry=args.model_registry,
        default_model_id=args.default_model_id,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
