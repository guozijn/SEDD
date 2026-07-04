import torch

from sedd_mini.official_finetune import apply_lora, merged_lora_state_dict, trainable_parameter_count


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
