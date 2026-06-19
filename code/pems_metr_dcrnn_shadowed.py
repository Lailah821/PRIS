"""DCRNN shadowed-input variant for PEMS-BAY / METR-LA.

Paper: Li et al. (ICLR 2018) "Diffusion Convolutional Recurrent Neural Network".

Paper-strict architecture (capacity hidden_dim only):
- Diffusion convolution: bidirectional random walk on graph.
  DConv(x) = sum_{k=0..K-1} (P_fwd^k X W_fwd_k + P_bwd^k X W_bwd_k) + b.
- K = 2 (paper default — order of diffusion).
- DCGRU cell: GRU where the 3 linear gates (reset/update/candidate) are replaced by
  diffusion conv over the concatenated [x ; h_prev] node features.
- Stack `num_layers` DCGRU cells. Last-step hidden of target slot 0 → Linear → T_out.

Shadowed adaptation:
- Augment the 8×8 neighbor-only `adj_block` to 9×9 (target slot 0 connected to all
  valid neighbors, binary edges; self-loops added).
- P_fwd, P_bwd: row-normalize adj and adj.T per sample.
- Output extracted at slot 0 (target) only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _row_normalize_batched(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rowsum = adj.sum(dim=-1, keepdim=True).clamp_min(eps)
    return adj / rowsum


class _DiffusionConv(nn.Module):
    """Diffusion convolution (paper Eq.2).

    Input  : x (B, N, F_in), P_fwd (B, N, N), P_bwd (B, N, N).
    Output : (B, N, F_out).
    Uses K diffusion steps for each direction + identity (k=0) → 1 + 2*K supports.
    """

    def __init__(self, in_dim: int, out_dim: int, K: int = 2):
        super().__init__()
        self.K = int(K)
        n_supports = 1 + 2 * self.K
        self.weight = nn.Parameter(torch.empty(in_dim * n_supports, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        P_fwd: torch.Tensor,
        P_bwd: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, N, F), P_*: (B, N, N)
        outs = [x]
        cur = x
        for _ in range(self.K):
            cur = torch.einsum("bnm,bmf->bnf", P_fwd, cur)
            outs.append(cur)
        cur = x
        for _ in range(self.K):
            cur = torch.einsum("bnm,bmf->bnf", P_bwd, cur)
            outs.append(cur)
        h = torch.cat(outs, dim=-1)  # (B, N, F*(1+2K))
        return h @ self.weight + self.bias


class _DCGRUCell(nn.Module):
    """DCGRU cell (paper Sec.3.2). GRU with diffusion conv replacing linear gates."""

    def __init__(self, in_dim: int, hidden_dim: int, K: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_zr = _DiffusionConv(in_dim + hidden_dim, 2 * hidden_dim, K)
        self.gate_h = _DiffusionConv(in_dim + hidden_dim, hidden_dim, K)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        P_fwd: torch.Tensor,
        P_bwd: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, N, F_in), h_prev: (B, N, H)
        xh = torch.cat([x, h_prev], dim=-1)
        zr = torch.sigmoid(self.gate_zr(xh, P_fwd, P_bwd))
        z, r = torch.split(zr, self.hidden_dim, dim=-1)
        xh_r = torch.cat([x, r * h_prev], dim=-1)
        cand = torch.tanh(self.gate_h(xh_r, P_fwd, P_bwd))
        return (1.0 - z) * cand + z * h_prev


class DCRNNShadowed(nn.Module):
    def __init__(
        self,
        adj_block: torch.Tensor,
        top_n: int = 8,
        in_channels: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        out_len: int = 24,
        K: int = 2,
    ):
        super().__init__()
        self.top_n = int(top_n)
        self.n_slots = 1 + self.top_n
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.out_len = int(out_len)
        self.K = int(K)

        if not isinstance(adj_block, torch.Tensor):
            adj_block = torch.as_tensor(adj_block, dtype=torch.float32)
        self.register_buffer("adj_block", adj_block.float())

        cells = []
        c_in = in_channels
        for _ in range(num_layers):
            cells.append(_DCGRUCell(c_in, hidden_dim, K))
            c_in = hidden_dim
        self.cells = nn.ModuleList(cells)
        self.head = nn.Linear(hidden_dim, out_len)

    def _augmented_adj(self, sensor_id: torch.Tensor) -> torch.Tensor:
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
        # x: (B, T, 1+top_n, C). sensor_id: (B,) long.
        B, T, N, C = x.shape
        adj_aug = self._augmented_adj(sensor_id)            # (B, N, N)
        P_fwd = _row_normalize_batched(adj_aug)              # (B, N, N)
        P_bwd = _row_normalize_batched(adj_aug.transpose(-1, -2))  # (B, N, N)

        h_list = [x.new_zeros(B, N, self.hidden_dim) for _ in range(self.num_layers)]
        for t in range(T):
            inp = x[:, t, :, :]  # (B, N, C)
            for i, cell in enumerate(self.cells):
                h_new = cell(inp, h_list[i], P_fwd, P_bwd)
                h_list[i] = h_new
                inp = h_new
        h_target = h_list[-1][:, 0, :]                       # (B, H) — target slot
        return self.head(h_target)                            # (B, T_out)
