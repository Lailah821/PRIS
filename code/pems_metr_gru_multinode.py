"""Multinode GRU for shadowed prediction setup.

Input:  (B, T_in, 1+top_n, C)  slot 0 = target (speed=0 masked), slots 1.. = neighbors
Output: (B, T_out)             target sensor's future speed (normalized)

Flatten slots into a single feature axis: (B, T_in, (1+top_n)*C) → GRU → head.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultinodeGRU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_slots: int,
        out_len: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_channels * n_slots
        self.out_len = out_len
        self.gru = nn.GRU(
            input_size=self.in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, out_len)

    def forward(self, x: torch.Tensor, sensor_id=None) -> torch.Tensor:
        # x: (B, T_in, n_slots, C). sensor_id is accepted for shadowed-model
        # forward-signature parity but ignored by this baseline.
        B, T, N_slots, C = x.shape
        x_flat = x.reshape(B, T, N_slots * C)
        out, _ = self.gru(x_flat)
        return self.head(out[:, -1, :])  # (B, T_out)
