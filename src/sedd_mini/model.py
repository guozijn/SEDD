from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.reshape(-1).float()
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=sigma.device).float() / max(half - 1, 1)
        )
        args = torch.log1p(sigma)[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class SEDDTransformer(nn.Module):
    """A compact bidirectional Transformer that predicts log probability ratios."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.seq_len = int(config["seq_len"])
        self.vocab_size = int(config["vocab_size"])
        self.mask_id = int(config["mask_id"])
        self.pad_id = int(config["pad_id"])
        d_model = int(config["d_model"])
        n_heads = int(config["n_heads"])
        d_ff = int(config["d_ff"])
        n_layers = int(config["n_layers"])
        dropout = float(config["dropout"])

        self.token_emb = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_id)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, input_ids: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
        _, seq_len = input_ids.shape
        if seq_len > self.seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds configured seq_len {self.seq_len}")
        x = self.token_emb(input_ids)
        x = x + self.pos_emb[:, :seq_len]
        x = x + self.time_emb(sigma)[:, None, :]
        x = self.dropout(x)
        padding_mask = input_ids.eq(self.pad_id)
        x = self.blocks(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        return self.lm_head(x)


def build_model(config: dict[str, Any]) -> SEDDTransformer:
    return SEDDTransformer(config)
