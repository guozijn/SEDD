import torch

from sedd_mini.official_finetune import (
    apply_lora,
    build_parser,
    merged_lora_state_dict,
    trainable_parameter_count,
)


def test_lora_merge_exports_plain_linear_keys():
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    original_total = sum(param.numel() for param in model.parameters())

    replaced = apply_lora(
        model,
        rank=2,
        alpha=4.0,
        dropout=0.0,
        targets=["0", "2"],
    )
    state = merged_lora_state_dict(model)

    assert replaced == ["0", "2"]
    assert 0 < trainable_parameter_count(model) < original_total
    assert "0.weight" in state
    assert "2.weight" in state
    assert not any("lora_a" in key or "lora_b" in key or ".base." in key for key in state)


def test_official_sft_cli_defaults_match_notebook_artifacts():
    args = build_parser().parse_args([])

    assert args.model_path == "runs/arc_models/base/checkpoint_base.pt"
    assert args.train_path == "data/processed/official_arc_easy_train.pt"
    assert args.valid_path == "data/processed/official_arc_easy_valid.pt"
    assert args.out_dir == "runs/arc_models/arc_lora_sft"
    assert args.steps == 1000
    assert args.warmup_steps == 50
    assert args.eval_every == 50
    assert args.log_every == 25
    assert args.save_every == 0
    assert args.lora_rank == 8
