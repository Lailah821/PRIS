"""Multinode DLinear for shadowed prediction setup.

Architecture (Zeng et al. AAAI 2023, "DLinear"):
- series_decomp(x) = (seasonal, trend) where trend = MovingAvg(kernel=25)
- 2 channel-independent Linear(T_in → T_out): seasonal + trend
- output channel 0 = target sensor's predicted future speed (normalized)

Shadowed input shape (B, T_in, 1+top_n, C) is flattened along the slot×channel
axis to (B, T_in, (1+top_n)*C). DLinear's channel-independence runs in
parallel over all flattened channels. Only target-slot channel 0 (raw or
z-scored speed) is selected for the head output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad_left = (kernel_size - 1) // 2
        self.pad_right = kernel_size - 1 - self.pad_left
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        x_p = x.permute(0, 2, 1)  # (B, C, T)
        x_p = F.pad(x_p, (self.pad_left, self.pad_right), mode="replicate")
        trend = self.avg_pool(x_p).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class MultinodeDLinear(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_slots: int,
        seq_len: int,
        pred_len: int,
        kernel_size: int = 25,
        individual: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_slots = n_slots
        self.channels = in_channels * n_slots
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.decomp = _SeriesDecomp(kernel_size=kernel_size)
        if individual:
            self.lin_seasonal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(self.channels)]
            )
            self.lin_trend = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(self.channels)]
            )
        else:
            self.lin_seasonal = nn.Linear(seq_len, pred_len)
            self.lin_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor, sensor_id=None) -> torch.Tensor:
        # x: (B, T_in, n_slots, C) → flat (B, T_in, n_slots*C)
        B, T, N_slots, C = x.shape
        x_flat = x.reshape(B, T, N_slots * C)
        seasonal, trend = self.decomp(x_flat)
        s_p = seasonal.permute(0, 2, 1)  # (B, channels, T_in)
        t_p = trend.permute(0, 2, 1)
        if self.individual:
            s_out = torch.stack(
                [self.lin_seasonal[i](s_p[:, i, :]) for i in range(self.channels)], dim=1
            )
            t_out = torch.stack(
                [self.lin_trend[i](t_p[:, i, :]) for i in range(self.channels)], dim=1
            )
        else:
            s_out = self.lin_seasonal(s_p)
            t_out = self.lin_trend(t_p)
        out = s_out + t_out  # (B, channels, T_pred)
        # target slot=0, channel=0 → flattened index 0
        return out[:, 0, :]  # (B, T_out)
