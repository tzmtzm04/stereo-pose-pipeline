"""SmoothNet PyTorch 定义（仅 export_smoothnet_npz.py 使用）。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from core.geometry.smoothnet_model import SmoothNetConfig


class SmoothNetResBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        return x + identity


class SmoothNetTorch(nn.Module):
    def __init__(self, cfg: SmoothNetConfig, dropout: float = 0.5) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(cfg.window_size, cfg.hidden_size),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[
                SmoothNetResBlock(cfg.hidden_size, cfg.res_hidden_size, dropout=dropout)
                for _ in range(cfg.num_blocks)
            ]
        )
        self.decoder = nn.Linear(cfg.hidden_size, cfg.output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(torch.float32)
        if x.shape[2] != self.cfg.window_size:
            raise ValueError(f"window mismatch {x.shape[2]} vs {self.cfg.window_size}")
        x = self.encoder(x)
        x = self.res_blocks(x)
        return self.decoder(x)
