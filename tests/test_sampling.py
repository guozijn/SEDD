import torch

from sedd_mini.model import build_model
from sedd_mini.sampling import sample_response
from sedd_mini.tokenizer import ByteTokenizer


def test_sampling_returns_text():
    tokenizer = ByteTokenizer()
    model = build_model(
        {
            "seq_len": 32,
            "vocab_size": tokenizer.vocab_size,
            "mask_id": tokenizer.mask_id,
            "pad_id": tokenizer.pad_id,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "d_ff": 64,
            "dropout": 0.0,
        }
    )
    text = sample_response(
        model,
        tokenizer,
        "Hi",
        max_new_tokens=8,
        steps=2,
        temperature=1.0,
        top_k=10,
        top_p=1.0,
        seq_len=32,
        device=torch.device("cpu"),
    )
    assert isinstance(text, str)
