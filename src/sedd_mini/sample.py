from __future__ import annotations

import argparse

from .backend import GenerationParams, create_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample from SEDD backends.")
    parser.add_argument("--backend", choices=["mini", "official"], default="official")
    parser.add_argument("--checkpoint", default="", help="Mini backend checkpoint path.")
    parser.add_argument("--model-path", default="louaaron/sedd-small", help="Official HF model path.")
    parser.add_argument(
        "--official-repo",
        default="external/Score-Entropy-Discrete-Diffusion",
        help="Path to cloned official SEDD repo.",
    )
    parser.add_argument("--prompt", default="Explain score entropy.")
    parser.add_argument("--infill", default="", help="Text containing [MASK] for infilling.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=23)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend = create_backend(
        args.backend,
        checkpoint=args.checkpoint,
        model_path=args.model_path,
        official_repo=args.official_repo,
        device_name=args.device,
        seed=args.seed,
    )
    params = GenerationParams(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    if args.infill:
        text = backend.infill(args.infill, params)
    else:
        text = backend.generate(args.prompt, params)
    print(text)


if __name__ == "__main__":
    main()
