"""Graph-WaveNet shadowed-input variant for PEMS-BAY / METR-LA.

Paper: Wu et al. (IJCAI 2019) "Graph WaveNet for Deep Spatial-Temporal Graph Modeling".

Paper-strict architecture (capacity unchanged):
- 4 blocks × 2 layers = 8 gated TCN layers, kernel_size=2.
- Dilations within block reset: [1, 2, 1, 2, 1, 2, 1, 2].
- Receptive field = sum(d*(k-1)) + 1 = 13.  Input padded to 13 from T_in=12.
- residual_channels = dilation_channels = 32 (paper default).
- skip_channels = 256, end_channels = 512.
- Per-layer: filter(tanh) × gate(sigmoid) → dropout → GCN → residual + BN.
- GCN supports: P_fwd + P_bwd + adaptive (3 supports, K_diffusion=2, identity skip).
- Adaptive adj A_adp = softmax(ReLU(E1 @ E2)).
- End: ReLU(skip_sum) → end_conv1 → ReLU → end_conv2 → (B, T_out, N).

Shadowed adaptation:
- Augment the 8×8 neighbor-only `adj_block` to a 9×9 graph by inserting target slot 0
  with binary edges to all valid neighbors. P_fwd / P_bwd row-normalized per sample.
- Output extracted at slot 0 (target) only → (B, T_out).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _row_normalize_batched(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-stochastic transition. adj: (B, N, N) → (B, N, N)."""
    rowsum = adj.sum(dim=-1, keepdim=True).clamp_min(eps)
    return adj / rowsum


class _GraphConvBatched(nn.Module):
    """Diffusion GCN with per-sample supports (batched).

    Supports: P_fwd (B,N,N), P_bwd (B,N,N), A_adp (N,N).
    For each support, accumulate K_order random-walk powers; concat with identity.
    """

    def __init__(self, c_in: int, c_out: int, k_order: int = 2, n_supports: int = 3):
        super().__init__()
        self.k_order = int(k_order)
        self.n_supports = int(n_supports)
        total_in = c_in * (1 + n_supports * k_order)
        self.mlp = nn.Conv2d(total_in, c_out, kernel_size=(1, 1))

    def forward(
        self,
        x: torch.Tensor,
        P_fwd: torch.Tensor,
        P_bwd: torch.Tensor,
        A_adp: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, C, N, T); P_fwd/P_bwd: (B, N, N); A_adp: (N, N)
        outs = [x]
        # batched supports
        for sup in [P_fwd, P_bwd]:
            cur = x
            for _ in range(self.k_order):
                cur = torch.einsum("bnm,bcmt->bcnt", sup, cur)
                outs.append(cur)
        # static support (broadcast over batch)
        cur = x
        for _ in range(self.k_order):
            cur = torch.einsum("nm,bcmt->bcnt", A_adp, cur)
            outs.append(cur)
        h = torch.cat(outs, dim=1)
        return self.mlp(h)


class _GWNetLayer(nn.Module):
    def __init__(
        self,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        k_order: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = float(dropout)
        self.filter_conv = nn.Conv2d(
            residual_channels, dilation_channels,
            kernel_size=(1, kernel_size), dilation=(1, dilation),
        )
        self.gate_conv = nn.Conv2d(
            residual_channels, dilation_channels,
            kernel_size=(1, kernel_size), dilation=(1, dilation),
        )
        self.gcn = _GraphConvBatched(
            dilation_channels, residual_channels,
            k_order=k_order, n_supports=3,
        )
        self.skip_conv = nn.Conv2d(
            dilation_channels, skip_channels, kernel_size=(1, 1)
        )
        self.bn = nn.BatchNorm2d(residual_channels)

    def forward(
        self,
        x: torch.Tensor,
        P_fwd: torch.Tensor,
        P_bwd: torch.Tensor,
        A_adp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f = torch.tanh(self.filter_conv(x))
        g = torch.sigmoid(self.gate_conv(x))
        h = f * g
        h = F.dropout(h, self.dropout, training=self.training)
        skip = self.skip_conv(h)
        h2 = self.gcn(h, P_fwd, P_bwd, A_adp)
        T_new = h2.shape[-1]
        x_cropped = x[..., -T_new:]
        out = self.bn(h2 + x_cropped)
        return out, skip


class GWNetShadowed(nn.Module):
    def __init__(
        self,
        adj_block: torch.Tensor,
        top_n: int = 8,
        in_channels: int = 1,
        hidden_dim: int = 64,  # unused; paper-strict residual=32, kept for API
        out_len: int = 24,
        input_len: int = 12,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        n_blocks: int = 4,
        layers_per_block: int = 2,
        adaptive_k: int = 10,
        k_order: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.top_n = int(top_n)
        self.n_slots = 1 + self.top_n
        self.in_channels = int(in_channels)
        self.out_len = int(out_len)
        self.input_len = int(input_len)
        self.kernel_size = int(kernel_size)
        self.n_blocks = int(n_blocks)
        self.layers_per_block = int(layers_per_block)
        self.k_order = int(k_order)
        self.dropout = float(dropout)

        if not isinstance(adj_block, torch.Tensor):
            adj_block = torch.as_tensor(adj_block, dtype=torch.float32)
        self.register_buffer("adj_block", adj_block.float())

        self.start_conv = nn.Conv2d(in_channels, residual_channels, kernel_size=(1, 1))

        # adaptive node embeddings
        self.E1 = nn.Parameter(torch.randn(self.n_slots, adaptive_k) * 0.01)
        self.E2 = nn.Parameter(torch.randn(adaptive_k, self.n_slots) * 0.01)

        # paper-strict 4×2 layers, dilations reset within block: [1,2,1,2,...]
        self.layers = nn.ModuleList()
        receptive_field = 1
        for _ in range(n_blocks):
            for layer_i in range(layers_per_block):
                dilation = 2 ** layer_i  # 1, 2 within block
                self.layers.append(
                    _GWNetLayer(
                        residual_channels=residual_channels,
                        dilation_channels=dilation_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        k_order=k_order,
                        dropout=dropout,
                    )
                )
                receptive_field += dilation * (kernel_size - 1)
        self.receptive_field = int(receptive_field)

        self.end_conv1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv2 = nn.Conv2d(end_channels, out_len, kernel_size=(1, 1))

    def _augmented_adj(self, sensor_id: torch.Tensor) -> torch.Tensor:
        """(B, 1+top_n, 1+top_n) — target slot 0 connected to valid neighbors (binary).
        Self-loops on target + valid neighbors. Symmetric.
        """
        B = sensor_id.shape[0]
        nbr_adj = self.adj_block[sensor_id]  # (B, top_n, top_n)
        nbr_valid = (nbr_adj.sum(dim=-1) > 0).float()  # (B, top_n)
        I_n = torch.eye(self.top_n, device=nbr_adj.device).unsqueeze(0).expand(B, -1, -1)
        nbr_with_self = nbr_adj + I_n * nbr_valid.unsqueeze(-1)
        N = self.n_slots
        adj_aug = nbr_adj.new_zeros(B, N, N)
        adj_aug[:, 0, 0] = 1.0
        adj_aug[:, 0, 1:] = nbr_valid
        adj_aug[:, 1:, 0] = nbr_valid
        adj_aug[:, 1:, 1:] = nbr_with_self
        return adj_aug

    def forward(self, x: torch.Tensor, sensor_id: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, 1+top_n, C). sensor_id: (B,) long.
        B, T, N, C = x.shape
        adj_aug = self._augmented_adj(sensor_id)            # (B, N, N)
        P_fwd = _row_normalize_batched(adj_aug)              # (B, N, N)
        P_bwd = _row_normalize_batched(adj_aug.transpose(-1, -2))  # (B, N, N)
        A_adp = F.softmax(F.relu(self.E1 @ self.E2), dim=1)  # (N, N)

        x_p = x.permute(0, 3, 2, 1)                          # (B, C, N, T_in)
        # left-pad time to receptive_field if T_in < receptive_field
        if T < self.receptive_field:
            pad_len = self.receptive_field - T
            x_p = F.pad(x_p, (pad_len, 0))

        h = self.start_conv(x_p)
        skip_accum = None
        for layer in self.layers:
            h, skip = layer(h, P_fwd, P_bwd, A_adp)
            if skip_accum is None:
                skip_accum = skip
            else:
                T_new = skip.shape[-1]
                skip_accum = skip_accum[..., -T_new:] + skip

        out = F.relu(skip_accum)
        out = F.relu(self.end_conv1(out))                    # (B, end_ch, N, T_new)
        out = out[..., -1:]                                  # (B, end_ch, N, 1)
        out = self.end_conv2(out)                            # (B, T_out, N, 1)
        out = out.squeeze(-1)                                # (B, T_out, N)
        return out[:, :, 0]                                  # (B, T_out) — target slot
