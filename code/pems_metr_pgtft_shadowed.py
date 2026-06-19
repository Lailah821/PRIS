"""PGTFT shadowed-input full dispatcher for PEMS-BAY / METR-LA.

11 mar 77 PGTFT ablation variants ported to shadowed setup + 3 adj-fusion lite
variants. Input format identical to PGGRUShadowed: (B, T, 1+top_n, C) with target
self-mask + sensor_id for adj_block lookup.

mar 77 ablations supported (from `1) for_paper_work/C0_C3_matrix_fillnan.md`):
  - paper           : full PGTFT body (SoftGCN + VSN + attn_grn + peak_grn + gate_block)
  - r1_1_hskip      : paper + horizon-skip decay (w_max=0.3, tau=60)
  - r2_a_l6a        : paper - peak_grn - dist_adj
  - r2_b_l6b        : r2_a + bypass VSN / attn_grn / gate_block
  - r2_c1           : r2_b + TCN (replace SoftGCN)
  - r2_c3           : r2_b + SoftGCN + TCN (additive)
  - r2_c4           : r2_b + target-node-select (slot 1 = strongest neighbor)
  - r3_delta        : r2_c1 + TCN dilations [1,1,2,2]
  - r3_lstgf        : r2_b + LSTGF (γ-fusion 2-stream)
  - r3_delta_gelu   : r3_delta + GELU activation
  - r3_delta_gelu_peak : r3_delta_gelu + per-layer peak gate (internal, no ext feature)

Shadowed-NA mar 77 variants (skipped):
  - R1-A swap : requires peak feature swap — no peak in shadowed
  - R1-D lite : R1-A + R1.1 stack — same caveat

Adapter notes for shadowed setup:
  - target slot 0 = self-masked (speed=0). mar 77 X_past[:,-1,0] references
    (anchor/horizon-skip) become slot 1 (strongest neighbor's last speed) here.
  - No X_future / static_cat / static_real / peak_feature. peak_grn becomes
    self-conditioned (attn_out, attn_out). gate_block gating_vector = None.
  - 13 mar 77 ablations → 11 portable + 2 shadowed-NA.

Original adj-fusion lite variants kept for backward compatibility:
  - a (DCRNN-style), b (AGCRN-style), c (Hybrid) — GRN bypass + VSN + soft_gcn
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from PGTFT_arch import (
    SoftAttentionGCN,
    MiniVariableSelector,
    Interpreted_Multi_head_Attention,
    GRN,
    GateBlock,
    TCNBlock,
    LSTGFBlock,
    build_distribution_based_adjacency,
    fuse_adj_matrix,
)


class LSTMencoder(nn.Module):
    """Single LSTM encoder."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output, h_n


class PGTFTShadowed(nn.Module):
    def __init__(
        self,
        adj_block: torch.Tensor,
        top_n: int = 8,
        in_channels: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        out_len: int = 24,
        num_heads: int = 4,
        dropout: float = 0.1,
        # adjacency
        adj_alpha: float = 0.5,
        use_dist_adj: bool = True,
        use_adaptive_adj: bool = False,
        use_napl: bool = False,
        node_embed_dim: int = 10,
        # graph / temporal branch
        use_soft_gcn: bool = True,
        use_tcn: bool = False,
        tcn_num_layers: int = 4,
        tcn_kernel_size: int = 2,
        tcn_dilations: list | None = None,
        tcn_activation: str = "relu",
        tcn_use_peak_gate: bool = False,
        tcn_peak_alpha: float = 0.3,
        use_lstgf: bool = False,
        lstgf_K: int = 2,
        lstgf_R: int = 6,
        lstgf_gamma_init: float = 0.0,
        # node-axis reduction
        use_vsn: bool = True,
        use_target_node_select: bool = False,
        # GRN gates
        use_attn_grn: bool = True,
        use_peak_grn: bool = True,
        use_final_gate_grn: bool = True,
        # horizon-skip
        use_horizon_skip: bool = False,
        hskip_w_max: float = 0.3,
        hskip_tau: float = 60.0,
    ):
        super().__init__()
        self.top_n = int(top_n)
        self.n_slots = 1 + self.top_n
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.out_len = int(out_len)
        self.num_heads = int(num_heads)
        self.adj_alpha = float(adj_alpha)
        self.use_dist_adj = bool(use_dist_adj)
        self.use_adaptive_adj = bool(use_adaptive_adj)
        self.use_napl = bool(use_napl)
        self.use_soft_gcn = bool(use_soft_gcn)
        self.use_tcn = bool(use_tcn)
        self.use_lstgf = bool(use_lstgf)
        self.use_vsn = bool(use_vsn)
        self.use_target_node_select = bool(use_target_node_select)
        self.use_attn_grn = bool(use_attn_grn)
        self.use_peak_grn = bool(use_peak_grn)
        self.use_final_gate_grn = bool(use_final_gate_grn)
        self.use_horizon_skip = bool(use_horizon_skip)
        self.hskip_w_max = float(hskip_w_max)
        self.hskip_tau = float(hskip_tau)

        if not isinstance(adj_block, torch.Tensor):
            adj_block = torch.as_tensor(adj_block, dtype=torch.float32)
        self.register_buffer("adj_block", adj_block.float())

        H = self.hidden_dim
        C = self.in_channels
        G = self.top_n

        # neighbor branch
        if self.use_soft_gcn:
            self.graph_context = SoftAttentionGCN(
                input_dim=C, output_dim=H,
                use_napl=use_napl, num_nodes=G, node_embed_dim=node_embed_dim,
            )
        else:
            self.graph_context = None

        if self.use_tcn:
            self.past_tcn = TCNBlock(
                in_channels=C, hidden_channels=H,
                num_layers=tcn_num_layers, kernel_size=tcn_kernel_size, dropout=dropout,
                dilations=tcn_dilations, activation=tcn_activation,
                use_peak_gate=tcn_use_peak_gate, peak_alpha=tcn_peak_alpha,
            )
        else:
            self.past_tcn = None

        if self.use_lstgf:
            self.lstgf = LSTGFBlock(
                in_channels=C, out_channels=H,
                K=lstgf_K, R=lstgf_R, num_nodes=G,
            )
            self.lstgf_proj_c = nn.Linear(H, H)
            self.lstgf_proj_g = nn.Linear(H, H)
            self.lstgf_gamma = nn.Parameter(torch.tensor(float(lstgf_gamma_init)))
            self.lstgf_layer_norm = nn.LayerNorm(H)

        if not (self.use_soft_gcn or self.use_tcn or self.use_lstgf):
            self.neighbor_proj = nn.Linear(C, H)
        else:
            self.neighbor_proj = None

        # non-GCN target projection
        self.non_gcn_proj = nn.Linear(C, H)

        # node-axis reduction
        if self.use_vsn:
            self.mini_vsn = MiniVariableSelector(
                num_vars=self.n_slots, input_dim=H, hidden_dim=H,
            )

        # temporal encoder
        self.lstm_encoder = LSTMencoder(H, H, num_layers)

        # self-attention
        self.self_attn = Interpreted_Multi_head_Attention(
            query_dim=H, key_dim=H, value_dim=H, num_heads=num_heads,
        )

        # post-attention GRN (single-arg call → c=None)
        if self.use_attn_grn:
            self.grn_after_attn = GRN(
                input_dim=H, output_dim=H, context_dim=H, dropout=dropout,
            )

        # peak GRN (self-conditioned in shadowed: a=c=attn_out)
        if self.use_peak_grn:
            self.peak_grn = GRN(
                input_dim=H, output_dim=H, context_dim=H, dropout=dropout,
            )

        # final gate
        if self.use_final_gate_grn:
            self.gate_block = GateBlock(
                input_dim=H, context_dim=H, dropout=dropout,
            )
        else:
            self.attn_norm = nn.LayerNorm(H)

        # direct multi-step head
        self.dense = nn.Linear(H, self.out_len)

        # adaptive adj / NAPL params
        if self.use_adaptive_adj or self.use_napl:
            self.node_embed = nn.Parameter(torch.randn(G, node_embed_dim) * 0.01)
        if self.use_adaptive_adj:
            self.adj_fuse_logits = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor, sensor_id: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1+top_n, C). sensor_id: (B,) long.
        B, T, S, C = x.shape
        assert S == self.n_slots, f"slot dim {S} != 1+top_n={self.n_slots}"
        assert C == self.in_channels, (
            f"in_channel {C} != model in_channels={self.in_channels}"
        )

        target_x = x[:, :, 0:1, :]       # (B, T, 1, C)
        neighbor_x = x[:, :, 1:, :]      # (B, T, top_n, C)
        neighbor_speed = neighbor_x[..., 0]  # (B, T, top_n)

        # adjacency fuse
        hop_adj = self.adj_block[sensor_id]  # (B, G, G)
        if self.use_dist_adj:
            dist_adj = build_distribution_based_adjacency(neighbor_speed)
            if self.use_adaptive_adj:
                adp = F.softmax(
                    F.relu(self.node_embed @ self.node_embed.transpose(0, 1)),
                    dim=-1,
                )
                adp_b = adp.unsqueeze(0).expand(B, -1, -1)
                w = F.softmax(self.adj_fuse_logits, dim=0)
                adj = w[0] * hop_adj + w[1] * dist_adj + w[2] * adp_b
            else:
                adj = fuse_adj_matrix(hop_adj, dist_adj, alpha=self.adj_alpha)
        else:
            adj = hop_adj

        # graph / temporal branch
        if self.use_soft_gcn:
            adj_T = adj.unsqueeze(1).repeat(1, T, 1, 1).reshape(
                B * T, self.top_n, self.top_n,
            )
            nbr_in = neighbor_x.reshape(B * T, self.top_n, C)
            ne = self.node_embed if self.use_napl else None
            nbr_gcn = self.graph_context(nbr_in, adj_T, node_embed=ne)  # (BT, G, H)
            nbr_gcn = nbr_gcn.view(B, T, self.top_n, self.hidden_dim)
            if self.use_tcn:  # R2-C3 additive
                nbr_flat = neighbor_x.permute(0, 2, 1, 3).reshape(
                    B * self.top_n, T, C,
                )
                tcn_out = self.past_tcn(nbr_flat)  # (B*G, T, H)
                tcn_out = tcn_out.view(B, self.top_n, T, self.hidden_dim).permute(
                    0, 2, 1, 3,
                )
                nbr_gcn = nbr_gcn + tcn_out
        elif self.use_lstgf:
            nbr_gcn = self.lstgf(neighbor_x, adj)  # (B, T, G, H)
        elif self.use_tcn:
            nbr_flat = neighbor_x.permute(0, 2, 1, 3).reshape(
                B * self.top_n, T, C,
            )
            tcn_out = self.past_tcn(nbr_flat)
            nbr_gcn = tcn_out.view(B, self.top_n, T, self.hidden_dim).permute(
                0, 2, 1, 3,
            )
        else:
            nbr_gcn = self.neighbor_proj(neighbor_x)  # (B, T, G, H)

        # non-GCN target projection
        tgt_proj = self.non_gcn_proj(target_x)  # (B, T, 1, H)

        # concat: [target, neighbors]
        combined = torch.cat([tgt_proj, nbr_gcn], dim=2)  # (B, T, 1+G, H)

        # node-axis reduction
        if self.use_lstgf:
            # γ-fusion two-stream: LN(W_c·target + γ·W_g·strongest_neighbor)
            c_t = tgt_proj.squeeze(2)        # (B, T, H)
            g_t = nbr_gcn[:, :, 0, :]         # (B, T, H) strongest neighbor LSTGF out
            selected = self.lstgf_layer_norm(
                self.lstgf_proj_c(c_t) + self.lstgf_gamma * self.lstgf_proj_g(g_t)
            )
        elif self.use_vsn:
            selected, _ = self.mini_vsn(combined)  # (B, T, H)
        elif self.use_target_node_select:
            # shadowed-adapted: slot 1 = strongest neighbor (slot 0 = self-masked target)
            selected = combined[:, :, 1, :]
        else:
            selected = combined.mean(dim=2)

        # LSTM encoder
        lstm_out, _h_n = self.lstm_encoder(selected)      # (B, T, H)

        # self-attention
        attn_out, _aw = self.self_attn(lstm_out, lstm_out, lstm_out)  # (B, T, H)

        # post-attention GRN (single-arg call: c=None)
        if self.use_attn_grn:
            attn_out = self.grn_after_attn(attn_out)

        # peak GRN (self-conditioned)
        if self.use_peak_grn:
            attn_out = attn_out + self.peak_grn(attn_out, attn_out)

        # final gate
        if self.use_final_gate_grn:
            gated = self.gate_block(attn_out, None)
        else:
            gated = self.attn_norm(lstm_out + attn_out)

        # direct multi-step head
        last = gated[:, -1, :]              # (B, H)
        out = self.dense(last)              # (B, T_out)

        # horizon-skip (shadowed-adapted: strongest neighbor's last speed as anchor)
        if self.use_horizon_skip:
            t_arange = torch.arange(self.out_len, device=out.device, dtype=out.dtype)
            w_t = self.hskip_w_max * torch.exp(-t_arange / self.hskip_tau)  # (T_out,)
            anchor = x[:, -1, 1, 0]          # (B,) strongest neighbor's last speed
            out = (1.0 - w_t.unsqueeze(0)) * out + w_t.unsqueeze(0) * anchor.unsqueeze(1)

        return out
