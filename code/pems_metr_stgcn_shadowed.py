"""STGCN shadowed-input variant for PEMS-BAY / METR-LA.

Paper: Yu et al. (IJCAI 2018) "Spatio-Temporal Graph Convolutional Networks".

Paper-strict architecture (capacity unchanged):
- 2 ST-Conv blocks: channels (c_in, 32, 64) → (64, 32, 128), Chebyshev K=3.
- Per-block: TemporalGLU(k=3) → ChebConv(K=3) → TemporalGLU(k=3) + LayerNorm.
- Output TCN with kernel = T_after_blocks (squeeze time → 1), then per-node Linear.

Shadowed adaptation:
- Augment the 8×8 neighbor-only `adj_block` to a 9×9 graph by inserting target slot 0
  with binary edges to all neighbors (the target is by definition adjacent to all its
  top-N neighbors). Self-loops added before symmetric normalization.
- Output extracted at slot 0 (target) only.

Capacity caveat: paper hidden channels (32/64/128) are kept paper-strict; only the
output head is per-target so the model returns (B, T_out) instead of (B, T_out, N).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scaled_laplacian_batched(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute Chebyshev-ready scaled Laplacian for batched adjacency.

    L_hat = 2 L / lambda_max - I, where L = I - D^-0.5 A D^-0.5.
    adj: (B, N, N) symmetric, non-negative.
    Returns (B, N, N) float.
    """
    B, N, _ = adj.shape
    deg = adj.sum(dim=-1)  # (B, N)
    deg_inv_sqrt = torch.where(deg > eps, deg.pow(-0.5), torch.zeros_like(deg))
    A_norm = adj * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
    I = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(B, -1, -1)
    L = I - A_norm
    # eigvalsh is batched in torch
    eigvals = torch.linalg.eigvalsh(L)  # (B, N) ascending
    lambda_max = eigvals[:, -1].clamp_min(eps).view(B, 1, 1)
    return 2.0 * L / lambda_max - I


class _TemporalGLU(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_a = nn.Conv2d(c_in, c_out, kernel_size=(1, kernel_size), padding=0)
        self.conv_b = nn.Conv2d(c_in, c_out, kernel_size=(1, kernel_size), padding=0)
        if c_in != c_out:
            self.residual = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
        else:
            self.residual = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, N, T)
        crop = self.kernel_size - 1
        x_cropped = x[..., crop:]
        res = x_cropped if self.residual is None else self.residual(x_cropped)
        gated = self.conv_a(x) * torch.sigmoid(self.conv_b(x))
        return gated + res


class _ChebConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, K: int = 3):
        super().__init__()
        self.K = int(K)
        self.thetas = nn.Parameter(torch.empty(self.K, c_in, c_out))
        nn.init.xavier_uniform_(self.thetas)
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, x: torch.Tensor, L_hat: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, N, T) ; L_hat: (B, N, N)
        T_list = [x]
        if self.K > 1:
            T_list.append(torch.einsum("bnm,bcmt->bcnt", L_hat, x))
        for k in range(2, self.K):
            T_k = 2.0 * torch.einsum("bnm,bcmt->bcnt", L_hat, T_list[-1]) - T_list[-2]
            T_list.append(T_k)
        out = torch.einsum("bcnt,co->bont", T_list[0], self.thetas[0])
        for k in range(1, self.K):
            out = out + torch.einsum("bcnt,co->bont", T_list[k], self.thetas[k])
        return out + self.bias[None, :, None, None]


class _STConvBlock(nn.Module):
    def __init__(self, c_in: int, c_mid: int, c_out: int, kernel_t: int = 3, K: int = 3):
        super().__init__()
        self.tcn1 = _TemporalGLU(c_in, c_mid, kernel_size=kernel_t)
        self.gcn = _ChebConv(c_mid, c_mid, K=K)
        self.tcn2 = _TemporalGLU(c_mid, c_out, kernel_size=kernel_t)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(c_out)

    def forward(self, x: torch.Tensor, L_hat: torch.Tensor) -> torch.Tensor:
        h = self.tcn1(x)
        h = self.relu(self.gcn(h, L_hat))
        h = self.tcn2(h)
        h = h.permute(0, 2, 3, 1)
        h = self.layer_norm(h)
        return h.permute(0, 3, 1, 2)


class STGCNShadowed(nn.Module):
    def __init__(
        self,
        adj_block: torch.Tensor,
        top_n: int = 8,
        in_channels: int = 1,
        hidden_dim: int = 64,
        out_len: int = 24,
        input_len: int = 12,
        kernel_t: int = 3,
        K: int = 3,
        block1_mid: int = 32,
        block1_out: int = 64,
        block2_mid: int = 32,
        block2_out: int = 128,
    ):
        super().__init__()
        self.top_n = int(top_n)
        self.n_slots = 1 + self.top_n
        self.in_channels = int(in_channels)
        self.out_len = int(out_len)
        self.K = K

        if not isinstance(adj_block, torch.Tensor):
            adj_block = torch.as_tensor(adj_block, dtype=torch.float32)
        self.register_buffer("adj_block", adj_block.float())

        self.block1 = _STConvBlock(in_channels, block1_mid, block1_out, kernel_t, K)
        self.block2 = _STConvBlock(block1_out, block2_mid, block2_out, kernel_t, K)

        t_after = input_len - 4 * (kernel_t - 1)
        if t_after <= 0:
            raise ValueError(
                f"input_len({input_len}) too small for kernel_t({kernel_t})"
            )
        self.output_tcn = _TemporalGLU(block2_out, block2_out, kernel_size=t_after)
        self.head = nn.Linear(block2_out, out_len)

    def _augmented_adj(self, sensor_id: torch.Tensor) -> torch.Tensor:
        """Return (B, 1+top_n, 1+top_n) adjacency with target slot 0 connected to all
        valid neighbors (binary edges), plus self-loops. Symmetric.
        """
        B = sensor_id.shape[0]
        nbr_adj = self.adj_block[sensor_id]  # (B, top_n, top_n)
        # detect valid neighbor slots from non-zero rows in adj_block
        # nbr_valid: (B, top_n) — slot j is valid iff its row has any non-zero entry
        nbr_valid = (nbr_adj.sum(dim=-1) > 0).float()  # (B, top_n)
        I_n = torch.eye(self.top_n, device=nbr_adj.device).unsqueeze(0).expand(B, -1, -1)
        # add self-loop to valid neighbors
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
        B, T, S, C = x.shape
        adj_aug = self._augmented_adj(sensor_id)         # (B, N, N)
        L_hat = _scaled_laplacian_batched(adj_aug)        # (B, N, N)
        x_p = x.permute(0, 3, 2, 1)                       # (B, C, N, T)
        h = self.block1(x_p, L_hat)                       # (B, 64, N, T-4)
        h = self.block2(h, L_hat)                         # (B, 128, N, T-8)
        h = self.output_tcn(h)                            # (B, 128, N, 1)
        h = h.squeeze(-1)                                 # (B, 128, N)
        h_target = h[:, :, 0]                             # (B, 128) — target slot
        return self.head(h_target)                        # (B, T_out)
