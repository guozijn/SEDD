import torch

from sedd_mini.official_backend import _install_flash_attn_fallback


def test_flash_attn_fallback_shape():
    _install_flash_attn_fallback()
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.layers.rotary import apply_rotary_emb_qkv_

    qkv = torch.randn(6, 3, 2, 4)
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    out = flash_attn_varlen_qkvpacked_func(qkv, cu, 3, 0.0, causal=False)
    assert out.shape == (6, 2, 4)

    packed = torch.randn(1, 6, 3, 2, 4)
    cos = torch.randn(6, 2)
    sin = torch.randn(6, 2)
    rotated = apply_rotary_emb_qkv_(packed, cos, sin)
    assert rotated.shape == packed.shape
