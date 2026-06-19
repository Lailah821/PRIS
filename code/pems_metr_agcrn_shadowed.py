"""AGCRN shadowed-input variant for PEMS-BAY / METR-LA.

Paper: Bai et al. (NeurIPS 2020) "Adaptive Graph Convolutional Recurrent Network for
Traffic Forecasting".

Paper-strict architecture (capacity hidden_dim only; Cheb_K and num_layers paper):
- Node embedding E ∈ R(N, embed_dim) — here N = 1+top_n slots (slot-wise embedding
  rather than sensor-wise, since shadowed setup has a different neighbor set per target).
- Adaptive adjacency: A_hat = softmax(ReLU(E E^T)) ∈ R(N, N).
- Chebyshev expansion of A_hat with K=2 (paper default).
- Node Adaptive Parameter Learning (NAPL): per-node weight derived from E and a shared
  weight pool — W_n = E @ W_pool (per-node out-channel matrix).
- AGCRN cell: GRU where linear gates (reset/update/cand) are replaced by adaptive GCN.
- Stack `num_layers` AGCRN cells; the last-step output of target slot 0 is sent to a
  Linear head producing T_out future steps.

NOTE: AGCRN paper does not use a fixed adjacency at all — adaptive adj only. So the
shadowed setup's `adj_block` (8×8 sensor-specific) is intentionally not used here.
Slot embedding learns to assign weight to target vs. neighbors generically.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AGCN(nn.Module):
    """NAPL adaptive GCN block (paper Eq.4-5).

    Input  : x (B, N, C_in), node_embed (N, D), supports (K, N, N) — Cheb_K powers of A_hat.
    Output : (B, N, C_out).
    """

    def __init__(self, c_in: int, c_out: int, K: int, embed_dim: int):
        super().__init__()
        self.K = K
        # Shared weight pool: (embed_dim, K * c_in, c_out)
        self.weight_pool = nn.Parameter(torch.empty(embed_dim, K * c_in, c_out))
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, c_out))
        nn.init.xavier_uniform_(self.weight_pool)
        nn.init.zeros_(self.bias_pool)

    def forward(self, x: torch.Tensor, node_embed: torch.Tensor, supports: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C_in)
        # node_embed: (N, D), supports: (K, N, N)
        # Cheb expansion: stack [T_k @ x] along channel
        x_g = torch.einsum("knm,bmc->bnkc", supports, x)  # (B, N, K, C_in)
        B, N, K, C_in = x_g.shape
        x_g = x_g.reshape(B, N, K * C_in)
        # NAPL: per-node weight = E_n @ W_pool, per-node bias = E_n @ b_pool
        # W_n: (N, K*C_in, C_out), b_n: (N, C_out)
        W_n = torch.einsum("nd,dko->nko", node_embed, self.weight_pool)
        b_n = torch.einsum("nd,do->no", node_embed, self.bias_pool)
        out = torch.einsum("bnk,nko->bno", x_g, W_n) + b_n[None, :, :]
        return out


class _AGCRNCell(nn.Module):
    """Single AGCRN cell — GRU with adaptive GCN replacing each linear gate (paper Eq.6)."""

    def __init__(self, c_in: int, hidden_dim: int, K: int, embed_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # gates: candidate, update, reset — all GCN over (x ; h_prev) concat
        self.gate_zr = _AGCN(c_in + hidden_dim, 2 * hidden_dim, K, embed_dim)
        self.gate_h = _AGCN(c_in + hidden_dim, hidden_dim, K, embed_dim)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor,
                node_embed: torch.Tensor, supports: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C_in), h_prev: (B, N, H)
        xh = torch.cat([x, h_prev], dim=-1)  # (B, N, C_in+H)
        zr = torch.sigmoid(self.gate_zr(xh, node_embed, supports))  # (B, N, 2H)
        z, r = torch.split(zr, self.hidden_dim, dim=-1)
        xh_r = torch.cat([x, r * h_prev], dim=-1)
        cand = torch.tanh(self.gate_h(xh_r, node_embed, supports))  # (B, N, H)
        return (1.0 - z) * cand + z * h_prev


class AGCRNShadowed(nn.Module):
    def __init__(
        self,
        top_n: int = 8,
        in_channels: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        out_len: int = 24,
        embed_dim: int = 10,
        K: int = 2,
    ):
        super().__init__()
        self.top_n = int(top_n)
        self.n_slots = 1 + self.top_n
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.out_len = int(out_len)
        self.embed_dim = int(embed_dim)
        self.K = int(K)

        # slot embedding (target slot 0 + top_n neighbors)
        self.node_embed = nn.Parameter(torch.randn(self.n_slots, embed_dim) * 0.05)

        cells = []
        c_in = in_channels
        for _ in range(num_layers):
            cells.append(_AGCRNCell(c_in, hidden_dim, K, embed_dim))
            c_in = hidden_dim
        self.cells = nn.ModuleList(cells)
        # final head: target slot's last hidden → T_out
        self.head = nn.Linear(hidden_dim, out_len)

    def _supports(self) -> torch.Tensor:
        """Chebyshev expansion of adaptive adj A_hat. Returns (K, N, N)."""
        E = self.node_embed
        A_hat = F.softmax(F.relu(E @ E.t()), dim=-1)  # (N, N)
        I_n = torch.eye(self.n_slots, device=E.device, dtype=E.dtype)
        supports = [I_n, A_hat]
        for k in range(2, self.K):
            supports.append(2.0 * A_hat @ supports[-1] - supports[-2])
        return torch.stack(supports[: self.K], dim=0)  # (K, N, N)

    def forward(self, x: torch.Tensor, sensor_id: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1+top_n, C). sensor_id ignored (AGCRN paper has no fixed adj).
        B, T, N, C = x.shape
        supports = self._supports()
        # initialize hidden states for each layer
        h_list = [x.new_zeros(B, N, self.hidden_dim) for _ in range(self.num_layers)]
        for t in range(T):
            inp = x[:, t, :, :]  # (B, N, C)
            for i, cell in enumerate(self.cells):
                h_new = cell(inp, h_list[i], self.node_embed, supports)
                h_list[i] = h_new
                inp = h_new
        h_target = h_list[-1][:, 0, :]  # (B, H) target slot 0
        return self.head(h_target)
