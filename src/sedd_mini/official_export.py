from __future__ import annotations

import argparse
from pathlib import Path

from .official_backend import load_official_components, save_official_checkpoint
from .utils import get_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an official HF SEDD model to a local checkpoint.")
    parser.add_argument("--model-path", default="louaaron/sedd-small")
    parser.add_argument("--official-repo", default="external/Score-Entropy-Discrete-Diffusion")
    parser.add_argument("--out", default="runs/arc_models/base/checkpoint_base.pt")
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = get_device(args.device)
    model, _, _, base_model_path, step = load_official_components(
        args.model_path,
        repo_path=args.official_repo,
        device=device,
    )
    save_official_checkpoint(
        Path(args.out),
        model=model,
        base_model_path=base_model_path,
        step=step,
        extra={"stage": "base_export", "source_model_path": args.model_path},
    )
    print(args.out)


if __name__ == "__main__":
    main()
