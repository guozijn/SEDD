import torch

from sedd_mini.diffusion import score_entropy_loss
from sedd_mini.model import build_model


def tiny_config():
    return {
        "seq_len": 16,
        "vocab_size": 260,
        "mask_id": 259,
        "pad_id": 0,
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "d_ff": 64,
        "dropout": 0.0,
    }


def test_score_entropy_loss_backward():
    torch.manual_seed(0)
    model = build_model(tiny_config())
    input_ids = torch.randint(1, 100, (2, 16))
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    loss, metrics = score_entropy_loss(model, input_ids, loss_mask, mask_id=259)
    assert torch.isfinite(loss)
    assert metrics["active_tokens"] >= 0
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())
