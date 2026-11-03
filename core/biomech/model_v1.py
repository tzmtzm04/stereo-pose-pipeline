"""BioModule v1 — 轻量 vendored 实现（与 UTSA bio_module/model.py 一致）。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

CRITERIA_DIMS: dict[str, int] = {
    "active_torque_angle_scaling_function": 68,
    "active_torque_angular_velocity_scaling_function": 68,
    "dimensionless_acceleration": 34,
    "dimensionless_active_torque": 34,
    "dimensionless_ground_reaction": 12,
    "dimensionless_ideal_torque": 34,
    "dimensionless_instantaneous_power": 34,
    "dimensionless_marker": 60,
    "dimensionless_passive_torque": 34,
    "dimensionless_seat_reaction": 1,
    "dimensionless_speed": 40,
    "normalized_active_torque": 68,
    "seat": 1,
    "touch": 2,
}

BINARY_CRITERIA = frozenset({"touch", "seat"})


class _PosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1)])


class BioModuleV1(nn.Module):
    """输入 (B, W, 17, 3) root-centred 米制 pose → 14 criteria × W。"""

    def __init__(
        self,
        win: int,
        criteria_dims: dict[str, int] | None = None,
        d_model: int = 256,
        nhead: int = 8,
        nlayers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        criteria_dims = criteria_dims or CRITERIA_DIMS
        self.criteria_dims = criteria_dims
        self.win = win
        in_dim = 17 * 3

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = _PosEnc(d_model, max_len=win + 8, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=nlayers,
            norm=nn.LayerNorm(d_model),
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, dim),
                )
                for name, dim in criteria_dims.items()
            }
        )

    def forward(self, poses_3d: torch.Tensor) -> dict[str, torch.Tensor]:
        b, w, j, c = poses_3d.shape
        x = poses_3d.reshape(b, w, j * c)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return {name: head(x) for name, head in self.heads.items()}
