# tft_model.py

import platform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAttentionGCN(nn.Module):
    """SoftAttentionGCN with optional NAPL (Node Adaptive Parameter Learning).

    Round 1 Step 2b — AGCRN-style NAPL:
      Standard: out_i = W · weight_i  (all nodes share W)
      NAPL    : out_i = W_i · weight_i, where W_i = E_i · W_pool
                E ∈ R^{N×d_e}, W_pool ∈ R^{d_e × in × out}
                → W_i ∈ R^{in × out} differs per node, learned implicitly
                  via node embedding. Bias also node-specific (E·b_pool).
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 128,
        use_napl: bool = False,
        num_nodes: int = 6,
        node_embed_dim: int = 10,
    ) -> None:
        super().__init__()
        self.use_napl = use_napl
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim

        if use_napl:
            # AGCRN NAPL: W_pool ∈ R^{d_e × in × out}, b_pool ∈ R^{d_e × out}
            self.W_pool = nn.Parameter(
                torch.randn(node_embed_dim, input_dim, output_dim) * 0.01
            )
            self.b_pool = nn.Parameter(torch.zeros(node_embed_dim, output_dim))
        else:
            self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.SiLU()

    def forward(self, x, adj, node_embed=None):
        # x: (BT, G, in), adj: (BT, G, G)
        agg = torch.matmul(adj, x)             # (BT, G, in)
        agg = F.gelu(agg)
        weight = F.softmax(agg, dim=-1)        # (BT, G, in) — attention 흉내

        if self.use_napl:
            # node_embed: (G, d_e) — TFTModel 에서 전달
            # W_per_node: (G, in, out), b_per_node: (G, out)
            W_per_node = torch.einsum("nd,dio->nio", node_embed, self.W_pool)
            b_per_node = node_embed @ self.b_pool                # (G, out)
            out = torch.einsum("bni,nio->bno", weight, W_per_node) + b_per_node
        else:
            out = self.linear(weight)
        return self.activation(out)


def compute_distribution_features(x):
    """
    x: (B, T, N) - 시간축 기준 입력 시계열
    return: (B, N, 4) - mean, std, skew, kurt
    """
    mean = x.mean(dim=1)
    std = x.std(dim=1) + 1e-6
    skew = ((x - mean.unsqueeze(1)) ** 3).mean(dim=1) / std.pow(3)
    kurt = ((x - mean.unsqueeze(1)) ** 4).mean(dim=1) / std.pow(4)
    return torch.stack([mean, std, skew, kurt], dim=-1)  # (B, N, 4)

def build_distribution_based_adjacency(x, similarity='cosine'):
    """
    x: (B, T, N) → returns adj: (B, N, N)
    """
    B, T, N = x.shape
    dist_feat = compute_distribution_features(x)  # (B, N, 4)

    if similarity == 'cosine':
        # normalize
        normed_feat = F.normalize(dist_feat, dim=-1)  # (B, N, 4)
        adj = torch.matmul(normed_feat, normed_feat.transpose(1, 2))  # (B, N, N)
        return adj
    else:
        raise NotImplementedError(f"Similarity metric {similarity} not implemented")
    
def fuse_adj_matrix(hop_adj, dist_adj, alpha=0.5):
    hop = alpha * hop_adj
    dist = (1-alpha) * dist_adj
    return hop + dist


class MixStyle(nn.Module):
    """MixStyle (Zhou et al., ICLR 2021).

    Round 2 Step (B-1): batch 내 다른 sample 의 (μ, σ) 를 섞어 style 다양성 주입.
    fair-comparison 준수: training-time only. inference 시 identity (self.training=False).

    Input:  x ∈ (B, T, N, H) — combined GCN+non-GCN feature
    Output: x' ∈ (B, T, N, H), 동일 shape

    동작:
      1. T 축 기준 per-sample 통계 (μ, σ) 계산: shape (B, 1, N, H)
      2. 같은 batch 의 다른 index permutation 으로 (μ', σ') 추출
      3. λ ~ Beta(α, α), batch 별 mixing coefficient
      4. x_norm = (x - μ) / σ
      5. μ_mix = λ·μ + (1-λ)·μ',  σ_mix = λ·σ + (1-λ)·σ'
      6. out = x_norm · σ_mix + μ_mix
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self._beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if not self.training:
            return x
        if torch.rand(1).item() > self.p:
            return x
        B = x.size(0)
        if B < 2:
            return x

        mu = x.mean(dim=1, keepdim=True)                      # (B, 1, N, H)
        sig = x.std(dim=1, keepdim=True) + self.eps           # (B, 1, N, H)
        x_norm = (x - mu) / sig

        perm = torch.randperm(B, device=x.device)
        mu2 = mu[perm]
        sig2 = sig[perm]

        lam = self._beta.sample((B, 1, 1, 1)).to(x.device)    # (B, 1, 1, 1)
        mu_mix = lam * mu + (1.0 - lam) * mu2
        sig_mix = lam * sig + (1.0 - lam) * sig2

        return x_norm * sig_mix + mu_mix


class GRN(nn.Module):
    def __init__(self, input_dim, output_dim, context_dim, dropout):
        super(GRN, self).__init__()
        self.w2 = nn.Linear(input_dim, output_dim)
        self.w3 = nn.Linear(context_dim, output_dim)
        self.b2 = nn.Parameter(torch.zeros(output_dim))

        self.w1 = nn.Linear(input_dim, output_dim)
        self.b1 = nn.Parameter(torch.zeros(output_dim))

        self.w4 = nn.Linear(output_dim, output_dim)
        self.w5 = nn.Linear(output_dim, output_dim)
        self.b4 = nn.Parameter(torch.zeros(output_dim))
        self.b5 = nn.Parameter(torch.zeros(output_dim))

        self.elu = nn.ELU()
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(p=dropout) # 0.15


    def forward(self, a, c=None):
        b1 = self.b1.view(1, 1, -1).expand(a.size(0), a.size(1), -1)
        b2 = self.b2.view(1, 1, -1).expand(a.size(0), a.size(1), -1)
        b4 = self.b4.view(1, 1, -1).expand(a.size(0), a.size(1), -1)
        b5 = self.b5.view(1, 1, -1).expand(a.size(0), a.size(1), -1)

        if c is not None:
            if c.shape[1] != a.shape[1]:
                #c = c[:, :a.shape[1], :]
                c = c.unsqueeze(1).expand(-1, a.size(1), -1)  # (B, T, D)
            elif c.dim() == 3 and c.size(1) == 1:  # (B, 1, D)
                c = c.expand(-1, a.size(1), -1)
            eta_2 = self.elu(self.w2(a) + self.w3(c) + b2)
        else:
            eta_2 = self.elu(self.w2(a) + b2)

        eta_1 = self.w1(a) + eta_2 + b1
        gamma = eta_1
        glu = torch.sigmoid(self.w4(gamma) + b4) * (self.w5(gamma) + b5)
        output = self.layer_norm(a + self.dropout(glu))
        return output

class MiniVariableSelector(nn.Module):
    def __init__(self, num_vars, input_dim, hidden_dim):
        super().__init__()
        self.num_vars = num_vars
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 각 변수별 표현 학습 (예: 주변 도로 6개라면 6개의 Linear 레이어)
        self.var_transforms = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_vars)
        ])

        # 중요도 계산용 간단한 스코어 네트워크
        self.weight_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):  
        # x: (B, T, N, D) ← 예: (batch, time, num_vars, features per var)
        transformed = []
        for i in range(self.num_vars):
            var_x = x[:, :, i, :]  # (B, T, D)
            var_h = self.var_transforms[i](var_x)  # (B, T, H)
            transformed.append(var_h)

        var_stack = torch.stack(transformed, dim=2)  # (B, T, N, H)

        # 중요도 계산
        scores = self.weight_layer(var_stack).squeeze(-1)  # (B, T, N)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, N, 1)

        # 중요도 기반 가중 평균
        out = torch.sum(var_stack * weights, dim=2)  # (B, T, H)
        return out, weights.squeeze(-1)  # (B, T, H), (B, T, N)


class LSTMencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMencoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)


class LSTMdecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMdecoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        return output, (hidden, cell)


class Interpreted_Multi_head_Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads):
        super(Interpreted_Multi_head_Attention, self).__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.Wq = nn.Linear(query_dim, query_dim * num_heads)
        self.Wk = nn.Linear(key_dim, key_dim * num_heads)
        self.Wv = nn.Linear(value_dim, value_dim * num_heads)
        self.fc = nn.Linear(value_dim * num_heads, value_dim)

    def forward(self, query, key, value, pad_mask=None):
        batch_size = query.size(0)

        Q = self.Wq(query).view(batch_size, -1, self.num_heads, self.query_dim).transpose(1, 2)
        K = self.Wk(key).view(batch_size, -1, self.num_heads, self.key_dim).transpose(1, 2)
        V = self.Wv(value).view(batch_size, -1, self.num_heads, self.value_dim).transpose(1, 2)

        # Q, K shape: (batch, heads, seq_len, dim)
        dot_product = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.key_dim)  # (batch, heads, tgt_len, src_len)

        # 👇 pad_mask: (batch, src_len) → 확장 필요
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)
            dot_product = dot_product.masked_fill(pad_mask == 0, float('-inf'))

        attention_weights = F.softmax(dot_product, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.num_heads * self.value_dim)

        output = self.fc(attention_output)
        return output, attention_weights



class GateBlock(nn.Module):
    def __init__(self, input_dim, context_dim, dropout):
        super(GateBlock, self).__init__()
        self.grn = GRN(input_dim, input_dim, context_dim, dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, input_vector, gating_vector):
        gated_output = self.grn(input_vector, gating_vector)
        output = self.layer_norm(input_vector + gated_output)
        return output


# ---------------- R2-C1 : dilated TCN block (Graph-WaveNet style) ----------------
class GatedDilatedConv1d(nn.Module):
    """Causal gated dilated conv1d: out = tanh(W_f * x) ⊙ sigmoid(W_g * x)."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size=kernel_size, dilation=dilation)
        self.dilation = int(dilation)
        self.kernel_size = int(kernel_size)

    def forward(self, x):
        # x: (B, C, T) — channels-first
        pad_left = self.dilation * (self.kernel_size - 1)
        x_pad = F.pad(x, (pad_left, 0))
        out = self.conv(x_pad)  # (B, 2C, T)
        filt, gate = out.chunk(2, dim=1)
        return torch.tanh(filt) * torch.sigmoid(gate)


class TCNBlock(nn.Module):
    """Graph-WaveNet style dilated TCN stack with optional peak-aware skip gating.

    - per-node temporal mixing (graph propagation 없음, R2-C1 = TCN-only ablation 용)
    - num_layers 개의 gated dilated causal conv1d, dilations=[1,2,4,...]
    - per-layer residual + skip → final activation + 1x1 conv

    R4 (2026-05-14):
      activation: "relu" (Graph-WaveNet original) / "elu" / "gelu" (skip_sum 비음수 clip 제거)
      use_peak_gate: per-layer learned peak relevance score (B, 1, T) 가 skip path 만
        scale (1 + α·peak_score). residual 은 untouched (안정성 보존).
      peak_alpha: peak amplification 강도 (default 0.3, 추천 sweep [0.1, 0.3, 0.5, 0.7]).

    Input/output shape: (B, T, C_in) → (B, T, C_out=hidden_channels)
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 num_layers: int = 4, kernel_size: int = 2, dropout: float = 0.1,
                 dilations: list[int] | None = None,
                 activation: str = "relu",
                 use_peak_gate: bool = False,
                 peak_alpha: float = 0.3):
        super().__init__()
        # R3-δ (2026-05-13): custom dilation list 지원.
        #   dilations=None → 기존 [2^0, 2^1, ..., 2^(num_layers-1)] (R2-C1 default)
        #   dilations=[1,1,2,2] → dense local TCN (R3-δ, receptive ≈ 6)
        if dilations is not None:
            dilations = [int(d) for d in dilations]
            num_layers = len(dilations)
        else:
            dilations = [2 ** i for i in range(num_layers)]
        self.dilations = list(dilations)
        self.activation = activation.lower()
        if self.activation not in ("relu", "elu", "gelu"):
            raise ValueError(f"Unsupported TCN activation: {activation}")
        self.use_peak_gate = bool(use_peak_gate)
        self.peak_alpha = float(peak_alpha)

        self.in_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.layers = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.peak_gates = nn.ModuleList()
        gate_hidden = max(hidden_channels // 2, 1)
        for dilation in dilations:
            self.layers.append(GatedDilatedConv1d(hidden_channels, dilation, kernel_size))
            self.residual_convs.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1))
            self.skip_convs.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1))
            if self.use_peak_gate:
                self.peak_gates.append(
                    nn.Sequential(
                        nn.Conv1d(hidden_channels, gate_hidden, kernel_size=1),
                        nn.GELU(),
                        nn.Conv1d(gate_hidden, 1, kernel_size=1),
                        nn.Sigmoid(),
                    )
                )
            else:
                self.peak_gates.append(nn.Identity())
        self.out_proj = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def _activate(self, x):
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "elu":
            return F.elu(x)
        return F.gelu(x)

    def forward(self, x, return_peak_scores: bool = False):
        # x: (B, T, C_in) → channels-first → TCN → channels-last
        x = x.transpose(1, 2)            # (B, C_in, T)
        x = self.in_proj(x)              # (B, C_hidden, T)
        skip_sum = None
        peak_scores: list[torch.Tensor] = []
        for layer, res_conv, skip_conv, peak_gate in zip(
            self.layers, self.residual_convs, self.skip_convs, self.peak_gates
        ):
            gated = layer(x)              # (B, C, T)
            if self.use_peak_gate:
                peak_score = peak_gate(gated)               # (B, 1, T)
                peak_scores.append(peak_score)
                sk_input = gated * (1.0 + self.peak_alpha * peak_score)
            else:
                sk_input = gated
            sk = skip_conv(sk_input)
            skip_sum = sk if skip_sum is None else skip_sum + sk
            x = x + res_conv(gated)       # residual path 는 untouched
            x = self.dropout(x)
        out = self._activate(skip_sum)
        out = self.out_proj(out)          # (B, C_hidden, T)
        out = out.transpose(1, 2)         # (B, T, C_hidden)
        if return_peak_scores and self.use_peak_gate:
            return out, torch.cat(peak_scores, dim=1)  # (B, L, T)
        return out


# ---------------- R3-LSTGF : Local Spatiotemporal Graph Filter ----------------
class LSTGFBlock(nn.Module):
    """Local Spatiotemporal Graph Filter (R3-LSTGF, 2026-05-13).

    g_t = Σ_{k=0..K} Σ_{τ=0..R} θ_{k,τ} P^k X_{t-τ}

    where:
      P = row-normalized adjacency (B, N, N)
      K = max graph hop (locality)
      R = max temporal lookback (frames)
      θ_{k,τ} = learnable kernel (C_in → C_out)

    핵심 motivation:
      "가까운 시간 + 가까운 공간" joint kernel. dilation-only TCN (R3-δ) NULL,
      hop-only SoftGCN ≈ TCN (R2-C3) TIED → unified spatial-temporal kernel 로
      두 축을 동시에 mix.

    Input: X (B, T, N, C_in), adj (B, N, N)
    Output: (B, T, N, C_out)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 K: int = 2, R: int = 6, num_nodes: int = 6):
        super().__init__()
        self.K = int(K)
        self.R = int(R)
        self.num_nodes = int(num_nodes)
        # θ_{k,τ}: (K+1) × (R+1) × in_channels × out_channels
        scale = 1.0 / max(1.0, float(in_channels)) ** 0.5
        self.theta = nn.Parameter(
            torch.randn(self.K + 1, self.R + 1, in_channels, out_channels) * scale
        )

    @staticmethod
    def _row_normalize(adj: torch.Tensor) -> torch.Tensor:
        # adj: (B, N, N) or (N, N) → row-stochastic
        row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return adj / row_sum

    def forward(self, X: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # X: (B, T, N, C_in), adj: (B, N, N) or (N, N)
        B, T, N, C_in = X.shape
        device = X.device
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)
        Pn = self._row_normalize(adj)  # (B, N, N)

        # P^k for k=0..K. P^0 = I (broadcast over batch).
        eye = torch.eye(N, device=device, dtype=X.dtype).unsqueeze(0).expand(B, -1, -1)
        P_powers = [eye]
        for _ in range(1, self.K + 1):
            P_powers.append(torch.bmm(P_powers[-1], Pn))

        # Pad past with zeros for τ-lookback up to R frames.
        pad = X.new_zeros(B, self.R, N, C_in)
        X_padded = torch.cat([pad, X], dim=1)  # (B, T+R, N, C_in)

        out_channels = self.theta.shape[-1]
        out = X.new_zeros(B, T, N, out_channels)
        for k in range(self.K + 1):
            P_k = P_powers[k]  # (B, N, N)
            for tau in range(self.R + 1):
                X_lag = X_padded[:, self.R - tau : self.R - tau + T]  # (B, T, N, C_in)
                # spatial: P^k @ X_lag along node axis
                Xspat = torch.einsum('bnm,btmc->btnc', P_k, X_lag)
                # channel mix: Xspat @ θ_{k,τ}
                out = out + torch.einsum('btnc,co->btno', Xspat, self.theta[k, tau])
        return out


class TFTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, target_dim, dropout, num_inputs, num_quantiles=3, static_cat_dim=None, static_real_dim=None,
                 use_adaptive_adj: bool = False, node_embed_dim: int = 10,
                 use_napl: bool = False,
                 use_mixstyle: bool = False, mixstyle_p: float = 0.5,
                 mixstyle_alpha: float = 0.1,
                 use_anchor_residual: bool = False, anchor_topk: int = 6,
                 anchor_clamp: float | None = None,
                 use_anchor_gate: bool = False, anchor_gate_bias_init: float = 0.0,
                 use_horizon_skip: bool = False, hskip_w_max: float = 0.3,
                 hskip_tau: float = 60.0,
                 use_peak_grn: bool = True, use_dist_adj: bool = True,
                 use_vsn: bool = True, use_attn_grn: bool = True,
                 use_final_gate_grn: bool = True,
                 use_soft_gcn: bool = True, use_tcn: bool = False,
                 tcn_num_layers: int = 4, tcn_kernel_size: int = 2,
                 tcn_dilations: list[int] | None = None,
                 tcn_activation: str = "relu",
                 tcn_use_peak_gate: bool = False,
                 tcn_peak_alpha: float = 0.3,
                 use_target_node_select: bool = False,
                 use_lstgf: bool = False,
                 lstgf_K: int = 2, lstgf_R: int = 6,
                 lstgf_gamma_init: float = 0.0,
                 strip_features: bool = False,
                 strip_aug: bool | None = None,
                 strip_static: bool | None = None):
        super(TFTModel, self).__init__()
        self.num_inputs = num_inputs

        # Input-cell ablation (Track A 4-cell, 2026-05-14 PGTFT port — PGGRU 패턴 동일):
        #   (False, False) = C3 full        — aug + static 둘 다 사용 (default)
        #   (True,  False) = C2 (1ch+static) — dynamic aug 0-mask, static 사용
        #   (False, True)  = C1 (aug only)   — dynamic aug 사용, static None
        #   (True,  True)  = C0 stripped    — 둘 다 제거 (PG-stripped fairness)
        # legacy `strip_features` (bool) 는 두 flag 모두 True 일 때만 True (alias).
        legacy_strip = bool(strip_features)
        self.strip_aug = bool(strip_aug) if strip_aug is not None else legacy_strip
        self.strip_static = bool(strip_static) if strip_static is not None else legacy_strip
        self.strip_features = self.strip_aug and self.strip_static
        # R2-C4 (2026-05-13): mean-pool 제거 + target-node-0 extraction
        #   use_vsn False, use_target_node_select True 일 때 x_combined[:, :, 0, :] 로 fixed.
        #   GCN-TFT / Graph-WaveNet 의 target-node preserving 과 fair comparison.
        self.use_target_node_select = bool(use_target_node_select)

        #self.lstm_encoder = LSTMencoder(input_dim, hidden_dim, num_layers)
        #self.lstm_decoder = LSTMdecoder(input_dim, hidden_dim, num_layers)
        self.lstm_encoder = LSTMencoder(hidden_dim, hidden_dim, num_layers)
        self.lstm_decoder = LSTMdecoder(hidden_dim, hidden_dim, num_layers)
        self.attention = Interpreted_Multi_head_Attention(
            query_dim=hidden_dim, key_dim=hidden_dim, value_dim=hidden_dim, num_heads=num_heads
        )
        # L6b (R2-B): paper-strict TFT 의 GRN 컴포넌트 3개 toggle.
        #   use_attn_grn=False         : grn_after_attention 우회 (PWFFN 위치 GRN 제거)
        #   use_final_gate_grn=False   : gate_block 의 GRN 제거, LN(input + static bias) 단순화
        if use_attn_grn:
            self.grn_after_attention = GRN(hidden_dim, hidden_dim, hidden_dim, dropout)
        else:
            self.grn_after_attention = None
        if use_final_gate_grn:
            self.gate_block = GateBlock(hidden_dim, hidden_dim, dropout)
            self.simple_final_ln = None
        else:
            self.gate_block = None
            self.simple_final_ln = nn.LayerNorm(hidden_dim)
        self.num_quantiles = num_quantiles
        self.dense = nn.Linear(hidden_dim, target_dim * num_quantiles)
        self.target_dim = target_dim

        # Static feature layers (embedding과 projection 모두 hidden_dim으로 설정)
        self.static_cat_embed = nn.Embedding(10, hidden_dim) if static_cat_dim is not None else None
        #self.static_real_layer = nn.Linear(static_real_dim, hidden_dim) if static_real_dim is not None else None
        #self.peak_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        # 기존 (단순 gate). L6a (R2-A): use_peak_grn=False 면 PGTFT 추가 GRN 우회.
        if use_peak_grn:
            self.peak_grn = GRN(hidden_dim, hidden_dim, context_dim=hidden_dim, dropout=dropout)
        else:
            self.peak_grn = None
        self.static_real_layer = nn.Linear(3, hidden_dim) if static_real_dim is not None else None
        self.mean_cong_proj = nn.Linear(1, hidden_dim)  # 평균 혼잡도 투영

        #self.mini_vsn = MiniVariableSelector(num_vars=input_dim, input_dim=1, hidden_dim=hidden_dim)
        # L6b (R2-B): use_vsn=False 면 VSN 우회 → num_inputs 차원 mean-pool (scale-invariant).
        if use_vsn:
            self.mini_vsn = MiniVariableSelector(num_vars=num_inputs, input_dim=hidden_dim, hidden_dim=hidden_dim)
        else:
            self.mini_vsn = None
        # R2-C1 (2026-05-13): SoftAttentionGCN conditional bypass + TCN replacement
        #   use_soft_gcn=False : SoftAttentionGCN 우회 (graph propagation 제거)
        #   use_tcn=True       : past sequence 에 dilated TCN 적용 (per-node, no graph)
        #                        future 는 simple Linear(1, H) projection (no temporal mixing).
        if use_soft_gcn:
            self.graph_context = SoftAttentionGCN(
                input_dim=1, output_dim=hidden_dim,
                use_napl=use_napl, num_nodes=6, node_embed_dim=node_embed_dim,
            )
        else:
            self.graph_context = None
        if use_tcn:
            self.past_tcn = TCNBlock(
                in_channels=1, hidden_channels=hidden_dim,
                num_layers=int(tcn_num_layers), kernel_size=int(tcn_kernel_size),
                dropout=dropout,
                dilations=list(tcn_dilations) if tcn_dilations is not None else None,
                activation=str(tcn_activation),
                use_peak_gate=bool(tcn_use_peak_gate),
                peak_alpha=float(tcn_peak_alpha),
            )
        else:
            self.past_tcn = None
        # future graph-substitute projection (use_soft_gcn=False 시에만 활용).
        # past 가 TCN/identity 인 경우에도 future 는 동일 projection 으로 통일.
        if not use_soft_gcn:
            self.future_node_proj = nn.Linear(1, hidden_dim)
        else:
            self.future_node_proj = None

        # init 내부에 추가 (non-GCN 항목 projection용)
        self.non_gcn_proj = nn.Linear(1, hidden_dim)

        # AGCRN-style adaptive adjacency (Round 1 Step 2a) + NAPL (Step 2b)
        # node_embed E ∈ R^{6×d_e} 는 두 컴포넌트가 공유
        # - 2a: A_adp = softmax(ReLU(E E^T)), 3-way fuse w_h·A_hop + w_d·A_dist + w_a·A_adp
        # - 2b: W_i = E_i · W_pool (graph_context 내부, NAPL)
        self.use_adaptive_adj = use_adaptive_adj
        self.use_napl = use_napl
        self.node_embed_dim = node_embed_dim
        if use_adaptive_adj or use_napl:
            self.node_embed = nn.Parameter(torch.randn(6, node_embed_dim) * 0.01)
        if use_adaptive_adj:
            self.adj_fuse_logits = nn.Parameter(torch.zeros(3))

        # Round 2 (B-1): MixStyle for domain-style augmentation (training-time only)
        self.use_mixstyle = use_mixstyle
        self.mixstyle = (
            MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if use_mixstyle else None
        )

        # Round 2 (B-2a): Anchor Residual head (mask-aware uniform mean of 6 neighbors' last-time congestion)
        self.use_anchor_residual = use_anchor_residual
        self.anchor_topk = int(anchor_topk)
        self.anchor_clamp = float(anchor_clamp) if anchor_clamp is not None else None

        # Round 2 (B-2b): Anchor Gate (convex combination)
        #   ŷ = (1-g)·anchor + g·learned_pred,   g ∈ (0,1) per timestep
        # gate proj: (B, T_future, hidden_dim) -> (B, T_future, 1) -> sigmoid
        # bias init: 0.0 → sigmoid=0.5 balanced. positive → start more toward learned.
        self.use_anchor_gate = use_anchor_gate
        if use_anchor_gate:
            self.anchor_gate_proj = nn.Linear(hidden_dim, 1)
            with torch.no_grad():
                self.anchor_gate_proj.bias.fill_(float(anchor_gate_bias_init))

        # Round 6.5 / R1.1: Horizon-decay self-skip
        # y[t] = (1 - w(t)) * model_pred[t] + w(t) * y_last,
        # w(t) = w_max * exp(-t / tau).  y_last = X_past[:, -1, 0] (closest single
        # neighbor, NOT k-road anchor → fair-comparison OK with §2.2 framework).
        self.use_horizon_skip = use_horizon_skip
        self.hskip_w_max = float(hskip_w_max)
        self.hskip_tau = float(hskip_tau)

        # L6a (R2-A): PGTFT 추가 컴포넌트 toggle (default True = 기존 PGTFT 동작 유지)
        #   use_peak_grn=False : post-attention peak_grn 우회 (paper-strict TFT 동일 path)
        #   use_dist_adj=False : fused adjacency 의 dist branch 제거, hop_adj only
        self.use_peak_grn = bool(use_peak_grn)
        self.use_dist_adj = bool(use_dist_adj)

        # L6b (R2-B): paper-strict TFT GRN 컴포넌트 3개 toggle.
        #   use_vsn=False            : VSN 우회 (mean-pool across num_inputs)
        #   use_attn_grn=False       : grn_after_attention 우회 (identity)
        #   use_final_gate_grn=False : gate_block 의 GRN 우회 (LN(input + static_context bias))
        self.use_vsn = bool(use_vsn)
        self.use_attn_grn = bool(use_attn_grn)
        self.use_final_gate_grn = bool(use_final_gate_grn)

        # R2-C1 (2026-05-13): SoftAttentionGCN ↔ TCN toggle.
        #   use_soft_gcn=False : SoftAttentionGCN 우회 (graph propagation off)
        #   use_tcn=True       : past sequence 에 dilated TCN (per-node, no graph)
        self.use_soft_gcn = bool(use_soft_gcn)
        self.use_tcn = bool(use_tcn)
        self.tcn_num_layers = int(tcn_num_layers)
        self.tcn_kernel_size = int(tcn_kernel_size)
        # R3-δ (2026-05-13): custom dilation list (None → default 2^i).
        #   예: [1,1,2,2] → dense local TCN (receptive ≈ 6).
        self.tcn_dilations = (
            [int(d) for d in tcn_dilations] if tcn_dilations is not None else None
        )
        # R3-δ-gelu / peak-gate (2026-05-14)
        self.tcn_activation = str(tcn_activation)
        self.tcn_use_peak_gate = bool(tcn_use_peak_gate)
        self.tcn_peak_alpha = float(tcn_peak_alpha)

        # R3-LSTGF (2026-05-13): Local Spatiotemporal Graph Filter + two-stream fusion.
        #   use_lstgf=True 면 past_graph branch 를 LSTGF 로 대체.
        #   두 stream (covariate c_t = non-graph mean / graph-temporal g_t = LSTGF[node 0]) 를
        #   z_t = LN(W_c c_t + γ W_g g_t) 로 fusion. γ init=0 (covariate stream 만으로 시작).
        self.use_lstgf = bool(use_lstgf)
        self.lstgf_K = int(lstgf_K)
        self.lstgf_R = int(lstgf_R)
        self.lstgf_gamma_init = float(lstgf_gamma_init)
        if self.use_lstgf:
            self.lstgf = LSTGFBlock(
                in_channels=1, out_channels=hidden_dim,
                K=self.lstgf_K, R=self.lstgf_R, num_nodes=6,
            )
            self.lstgf_proj_c = nn.Linear(hidden_dim, hidden_dim)
            self.lstgf_proj_g = nn.Linear(hidden_dim, hidden_dim)
            self.lstgf_gamma = nn.Parameter(torch.full((1,), float(lstgf_gamma_init)))
            self.lstgf_layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.lstgf = None
            self.lstgf_proj_c = None
            self.lstgf_proj_g = None
            self.lstgf_gamma = None
            self.lstgf_layer_norm = None


    def forward(self, X_past, X_future, static_cat=None, static_real=None, pad_mask=None, adj_matrix=None):
        B, T_past, N = X_past.shape
        T_future = X_future.shape[1]
        # Capture pad_mask before downstream blocks overwrite it (line ~451 sets pad_mask=None).
        pad_mask_input = pad_mask

        # ── Input-cell ablation (Track A 4-cell, 2026-05-14 PGTFT) ─────────────
        # encoder_cont = 14d = 6 neighbor + peak + 6 delta + one_hop_avg.
        # GCN slots (ch 0..5) = 6 neighbors (preprocessing L99 에서 target 제외).
        # strip_aug=True  → ch 6+ (peak + 6 delta + one_hop_avg) 0 으로 마스킹.
        # strip_static=True → static_cat / static_real 둘 다 None.
        if self.strip_aug and N > 6:
            X_past = X_past.clone()
            X_future = X_future.clone()
            X_past[:, :, 6:] = 0
            X_future[:, :, 6:] = 0
        if self.strip_static:
            static_cat = None
            static_real = None

        # 🔹 GCN 적용할 변수 인덱스 (예: 앞의 6개 변수)
        GCN_idx = list(range(6))  # 일반적으로 [0,1,2,3,4,5]
        G = len(GCN_idx)
        #"""
        # GCN에 쓸 adj_matrix는 처음부터 6개만 자르자!
        hop_adj_matrix = adj_matrix[:, :6, :6]  # (B, 6, 6)
        GCN_idx = list(range(6))  # 예: 앞의 6개 변수 사용
        if self.use_dist_adj:
            dist_adj_matrix = build_distribution_based_adjacency(X_past)  # (B, N, N)
            dist_adj_matrix = dist_adj_matrix[:, GCN_idx, :][:, :, GCN_idx]  # (B, 6, 6)
            if self.use_adaptive_adj:
                adp = F.softmax(F.relu(self.node_embed @ self.node_embed.transpose(0, 1)), dim=-1)  # (6, 6)
                adp_b = adp.unsqueeze(0).expand(B, -1, -1)                                          # (B, 6, 6)
                w = F.softmax(self.adj_fuse_logits, dim=0)                                          # (3,)
                adj_matrix = w[0] * hop_adj_matrix + w[1] * dist_adj_matrix + w[2] * adp_b
            else:
                adj_matrix = fuse_adj_matrix(hop_adj_matrix, dist_adj_matrix, alpha=0.5) #alpha가 커지면 hop_based 더 사용
        else:
            # L6a (R2-A): dist branch 제거 — hop_adj only (paper-strict 에서 graph 자체가 없으나
            # PGTFT 는 graph 유지 + dist 만 제거 = hop_adj 단독 사용).
            if self.use_adaptive_adj:
                adp = F.softmax(F.relu(self.node_embed @ self.node_embed.transpose(0, 1)), dim=-1)
                adp_b = adp.unsqueeze(0).expand(B, -1, -1)
                # 2-way fuse (hop + adp), dist 제외. logits 의 첫 두 항만 사용.
                w_full = F.softmax(self.adj_fuse_logits[:2] if self.adj_fuse_logits.numel() >= 2 else self.adj_fuse_logits, dim=0)
                adj_matrix = w_full[0] * hop_adj_matrix + w_full[1] * adp_b
            else:
                adj_matrix = hop_adj_matrix
        #"""

        #print("hop_adj[0]:", hop_adj_matrix[0])
        #print("dist_adj[0]:", dist_adj_matrix[0])
        #print("fused_adj[0]:", adj_matrix[0])


        # 2️⃣ 패드 마스크 적용
        if pad_mask is not None:
            gcn_pad_mask = pad_mask[:, GCN_idx]  # (B, G)
            adj_matrix = adj_matrix * (gcn_pad_mask.unsqueeze(1) * gcn_pad_mask.unsqueeze(2))  # (B, G, G)

        # 3️⃣ GCN 입력 준비 및 마스킹
        x_past_gcn_input = X_past[:, :, GCN_idx].unsqueeze(-1)  # (B, T_past, G, 1)
        x_future_gcn_input = X_future[:, :, GCN_idx].unsqueeze(-1)  # (B, T_future, G, 1)

        if pad_mask is not None:
            x_past_gcn_input = x_past_gcn_input * gcn_pad_mask.unsqueeze(1).unsqueeze(-1)  # (B, T_past, G, 1)
            x_future_gcn_input = x_future_gcn_input * gcn_pad_mask.unsqueeze(1).unsqueeze(-1)


        # NAPL 시 graph_context 가 node_embed 필요. None 이면 기존 동작.
        ne_for_gcn = self.node_embed if self.use_napl else None

        B, T_past, G, D = x_past_gcn_input.shape
        B, T_future, G, D = x_future_gcn_input.shape

        # --------- graph_context (past / future) ---------
        # 기존 path (use_soft_gcn=True, use_tcn=False) : SoftAttentionGCN(adj) 적용. = L6b.
        # R2-C1 (use_soft_gcn=False, use_tcn=True)    : graph 우회 + per-node TCN.
        # R2-C3 (use_soft_gcn=True, use_tcn=True)     : SoftAttentionGCN + TCN additive combine.
        #   past 는 두 branch (x_past_g + x_past_t), future 는 SoftAttentionGCN 만
        #   (calendar covariate 가 future 의 정보원, TCN 무의미).
        if self.use_soft_gcn and self.graph_context is not None:
            adj_past = adj_matrix.unsqueeze(1).repeat(1, T_past, 1, 1).reshape(B * T_past, G, G)
            x_past_reshaped = x_past_gcn_input.reshape(B * T_past, G, D)
            x_past_g = self.graph_context(x_past_reshaped, adj_past, node_embed=ne_for_gcn)  # (B*T_past, G, H)
            x_past_g = x_past_g.view(B, T_past, G, -1)  # (B, T_past, G, H)

            adj_future = adj_matrix.unsqueeze(1).repeat(1, T_future, 1, 1).reshape(B * T_future, G, G)
            x_future_reshaped = x_future_gcn_input.reshape(B * T_future, G, D)
            x_future_gcn = self.graph_context(x_future_reshaped, adj_future, node_embed=ne_for_gcn)  # (B*T_future, G, H)
            x_future_gcn = x_future_gcn.view(B, T_future, G, -1)  # (B, T_future, G, H)

            # R2-C3 : additive TCN branch on past.
            if self.use_tcn and self.past_tcn is not None:
                x_past_tcn_in = x_past_gcn_input.permute(0, 2, 1, 3).reshape(B * G, T_past, D)
                x_past_tcn_out = self.past_tcn(x_past_tcn_in)                  # (B*G, T_past, H)
                x_past_t = x_past_tcn_out.reshape(B, G, T_past, -1).permute(0, 2, 1, 3)  # (B, T_past, G, H)
                x_past_gcn = x_past_g + x_past_t  # additive (parameter-free combine)
            else:
                x_past_gcn = x_past_g
        else:
            # R2-C1 path. past : TCN (per-node, no graph) ; future : Linear (1, H) projection.
            # R3-LSTGF path. past : LSTGF (spatio-temporal joint kernel) ; future : Linear (1, H).
            if self.use_lstgf and self.lstgf is not None:
                # x_past_gcn_input : (B, T_past, G, 1), adj_matrix : (B, G, G)
                x_past_gcn = self.lstgf(x_past_gcn_input, adj_matrix)              # (B, T_past, G, H)
            elif self.use_tcn and self.past_tcn is not None:
                # (B, T_past, G, 1) → permute to (B, G, T_past, 1) → reshape (B*G, T_past, 1)
                x_past_tcn_in = x_past_gcn_input.permute(0, 2, 1, 3).reshape(B * G, T_past, D)
                x_past_tcn_out = self.past_tcn(x_past_tcn_in)                  # (B*G, T_past, H)
                x_past_gcn = x_past_tcn_out.reshape(B, G, T_past, -1).permute(0, 2, 1, 3)  # (B, T_past, G, H)
            else:
                # R2-C2 placeholder : use_soft_gcn=False AND use_tcn=False → identity-like Linear.
                x_past_gcn = self.future_node_proj(x_past_gcn_input)            # (B, T_past, G, H)
            # future 는 R2-C1/R3-LSTGF 모두 LSTM decoder + calendar covariate 가 주역. 단순 projection.
            x_future_gcn = self.future_node_proj(x_future_gcn_input)            # (B, T_future, G, H)

        # --------- GCN 외 나머지 변수 (ex: 피크, 평균, 기타 도로 등) ---------
        x_past_non_gcn = X_past[:, :, G:]  # (B, T_past, N-G)
        x_future_non_gcn = X_future[:, :, G:]  # (B, T_future, N-G)

        if pad_mask is not None:
            non_gcn_pad_mask = pad_mask[:, G:]  # (B, N-G)
            x_past_non_gcn = x_past_non_gcn * non_gcn_pad_mask.unsqueeze(1)
            x_future_non_gcn = x_future_non_gcn * non_gcn_pad_mask.unsqueeze(1)

        x_past_non_gcn = x_past_non_gcn.unsqueeze(-1)  # (B, T_past, N-G, 1)
        x_future_non_gcn = x_future_non_gcn.unsqueeze(-1)
        x_past_non_gcn = self.non_gcn_proj(x_past_non_gcn)  # (B, T_past, N-G, H)
        x_future_non_gcn = self.non_gcn_proj(x_future_non_gcn)  # (B, T_future, N-G, H)

        # --------- GCN 결과 + 나머지 입력 결합 ---------
        x_past_combined = torch.cat([x_past_gcn, x_past_non_gcn], dim=2)  # (B, T_past, N, H)
        x_future_combined = torch.cat([x_future_gcn, x_future_non_gcn], dim=2)  # (B, T_future, N, H)

        # --------- MixStyle (training-time only, identity at eval) ---------
        if self.mixstyle is not None:
            x_past_combined = self.mixstyle(x_past_combined)
            x_future_combined = self.mixstyle(x_future_combined)

        # --------- mini VSN / node-axis reduction ---------
        # L6b (R2-B): use_vsn=False 면 mean-pool across num_inputs (scale-invariant, no GRN).
        # R2-C4 (2026-05-13): use_target_node_select=True 면 target node 0 만 추출
        # (mean-pool 로 인한 target-specific graph-filtered signal 희석 제거).
        # R3-LSTGF (2026-05-14): two-stream fusion z = LN(W_c c_t + γ W_g g_t)
        #   c_t = covariate stream (non-graph slot mean, peak/mean/calendar swap 보존)
        #   g_t = graph-temporal stream (LSTGF or future_node_proj 의 target node 0)
        #   γ init = lstgf_gamma_init (default 0 → covariate-only at init).
        if self.use_lstgf and self.lstgf is not None:
            # past : c_t = non-graph mean, g_t = LSTGF[node 0]
            c_t_past = x_past_non_gcn.mean(dim=2)                  # (B, T_past, H)
            g_t_past = x_past_gcn[:, :, 0, :]                       # (B, T_past, H)
            x_past_selected = self.lstgf_layer_norm(
                self.lstgf_proj_c(c_t_past)
                + self.lstgf_gamma * self.lstgf_proj_g(g_t_past)
            )                                                       # (B, T_past, H)
            # future : c_t = non-graph mean, g_t = future_node_proj[node 0]
            c_t_future = x_future_non_gcn.mean(dim=2)              # (B, T_future, H)
            g_t_future = x_future_gcn[:, :, 0, :]                   # (B, T_future, H)
            x_future_selected = self.lstgf_layer_norm(
                self.lstgf_proj_c(c_t_future)
                + self.lstgf_gamma * self.lstgf_proj_g(g_t_future)
            )                                                       # (B, T_future, H)
        elif self.use_vsn and self.mini_vsn is not None:
            x_past_selected, _ = self.mini_vsn(x_past_combined)     # (B, T_past, H)
            x_future_selected, _ = self.mini_vsn(x_future_combined) # (B, T_future, H)
        elif self.use_target_node_select:
            x_past_selected = x_past_combined[:, :, 0, :]            # (B, T_past, H)
            x_future_selected = x_future_combined[:, :, 0, :]        # (B, T_future, H)
        else:
            x_past_selected = x_past_combined.mean(dim=2)            # (B, T_past, H)
            x_future_selected = x_future_combined.mean(dim=2)        # (B, T_future, H)

        # --------- LSTM ---------
        encoder_outputs, (hidden, cell) = self.lstm_encoder(x_past_selected)
        decoder_outputs, _ = self.lstm_decoder(x_future_selected, hidden, cell)

        # --------- Attention ---------
        pad_mask = None
        if pad_mask is not None:
            T_k = encoder_outputs.size(1)
            pad_mask_time = torch.ones((pad_mask.size(0), T_k), dtype=pad_mask.dtype, device=pad_mask.device)
        else:
            pad_mask_time = None
            #print("pad none")

        attention_output, _ = self.attention(
            decoder_outputs, encoder_outputs, encoder_outputs, pad_mask=pad_mask_time
        )
        # L6b (R2-B): use_attn_grn=False 면 PWFFN 위치 GRN 우회 (identity).
        if self.use_attn_grn and self.grn_after_attention is not None:
            attention_output = self.grn_after_attention(attention_output)

        # --------- Static context ---------
        if static_cat is not None:
            cat_embed = self.static_cat_embed(static_cat.squeeze(-1))  # (B, N, D)
            cat_embed = cat_embed.mean(dim=1)  # (B, D)
        if static_real is not None:
            base_static_real = static_real[:, :, :3]
            real_embed = self.static_real_layer(base_static_real.mean(dim=1))  # (B, D)
            mean_cong = static_real[:, :, 3].mean(dim=1, keepdim=True)  # (B, 1)
            mean_cong_proj = self.mean_cong_proj(mean_cong)  # (B, D)
            real_embed = real_embed + mean_cong_proj

        if static_cat is not None and static_real is not None:
            static_context = cat_embed + real_embed
        elif static_cat is not None:
            static_context = cat_embed
        elif static_real is not None:
            static_context = real_embed
        else:
            static_context = torch.zeros((B, attention_output.size(-1)), device=attention_output.device)

        # --------- Peak 강조 ---------
        # L6a (R2-A): use_peak_grn=False 면 attention_output 그대로 (paper-strict path).
        if self.use_peak_grn and self.peak_grn is not None:
            peak_enhancement = self.peak_grn(attention_output, attention_output)
            #attention_output = self.post_attention_addnorm(attention_output, peak_enhancement)
            attention_output = attention_output + peak_enhancement


        # --------- Gate + Dense ---------
        # L6b (R2-B): use_final_gate_grn=False 면 GRN 우회 → LN(input + static_context bias).
        if self.use_final_gate_grn and self.gate_block is not None:
            gated_output = self.gate_block(attention_output, static_context)
        else:
            # static_context: (B, H) → (B, 1, H) → broadcast over T_future
            gated_output = self.simple_final_ln(attention_output + static_context.unsqueeze(1))
        final_output = self.dense(gated_output)
        final_output = final_output.view(final_output.shape[0], final_output.shape[1], self.target_dim, self.num_quantiles)

        # --------- Anchor head (B-2a residual or B-2b gate) ---------
        # x_past[:, :, 0:6] = raw congestion of 6 neighbor roads (NOT target).
        # anchor = mean of last-time neighbor congestion (mask-aware).
        # B-2a (residual): ŷ = anchor + clamp(residual, ±v).
        # B-2b (gate)    : ŷ = (1-g(x))·anchor + g(x)·learned_pred,  g_t per timestep via sigmoid.
        if self.use_anchor_residual or self.use_anchor_gate:
            if pad_mask_input is not None:
                neigh_mask = pad_mask_input[:, :6].to(final_output.dtype)
            else:
                neigh_mask = torch.ones(B, 6, device=final_output.device, dtype=final_output.dtype)
            neigh_last = X_past[:, -1, :6].to(final_output.dtype)             # (B, 6)
            m_sum = neigh_mask.sum(-1, keepdim=True).clamp_min(1e-6)          # (B, 1)
            anchor = (neigh_last * neigh_mask).sum(-1, keepdim=True) / m_sum  # (B, 1)
            anchor_b = anchor.view(B, 1, 1, 1).expand_as(final_output)        # (B, T_f, target_dim, Q)

            if self.use_anchor_gate:
                # gated_output: (B, T_future, hidden_dim) — pre-dense feature
                gate_logits = self.anchor_gate_proj(gated_output)             # (B, T_f, 1)
                gate = torch.sigmoid(gate_logits).unsqueeze(-1)               # (B, T_f, 1, 1)
                final_output = (1.0 - gate) * anchor_b + gate * final_output
            else:
                residual = final_output
                if self.anchor_clamp is not None:
                    residual = residual.clamp(-self.anchor_clamp, self.anchor_clamp)
                final_output = anchor_b + residual

        # --------- Horizon-decay self-skip (Round 6.5 / R1.1) ---------
        if self.use_horizon_skip:
            y_last = X_past[:, -1, 0]                                          # (B,)
            if pad_mask_input is not None:
                valid0 = pad_mask_input[:, 0].to(final_output.dtype)           # (B,)
            else:
                valid0 = torch.ones(
                    B, device=final_output.device, dtype=final_output.dtype,
                )
            t_idx = torch.arange(
                T_future, device=final_output.device, dtype=final_output.dtype,
            )
            w = self.hskip_w_max * torch.exp(-t_idx / self.hskip_tau)          # (T_f,)
            w_b = w.view(1, T_future, 1, 1) * valid0.view(B, 1, 1, 1)          # (B, T_f, 1, 1)
            y_last_b = y_last.view(B, 1, 1, 1).expand_as(final_output)
            final_output = (1.0 - w_b) * final_output + w_b * y_last_b

        return final_output


class ReconstructionTFTModel(nn.Module):
    """
    Reconstruction PGTFT: 인코더 + Temporal Self-Attention으로 동일 시간대 시계열 복원.

    기존 TFTModel과의 차이:
    - 디코더 없음 (x_future 불필요)
    - Self-Attention: LSTM encoder output에 temporal self-attention 적용
    - Deterministic 단일 출력 (quantile 아님)
    - 출력: (B, T_past, target_dim)

    파이프라인:
      GCN + Non-GCN → VSN → LSTM Encoder → Self-Attention → GRN → Static GRN → Proj
    """
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, target_dim,
                 dropout, num_inputs, static_cat_dim=None, static_real_dim=None):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim

        # 인코더
        self.lstm_encoder = LSTMencoder(hidden_dim, hidden_dim, num_layers)
        self.mini_vsn = MiniVariableSelector(
            num_vars=num_inputs, input_dim=hidden_dim, hidden_dim=hidden_dim
        )
        self.graph_context = SoftAttentionGCN(input_dim=1, output_dim=hidden_dim)
        self.non_gcn_proj = nn.Linear(1, hidden_dim)

        # Temporal Self-Attention (TFTModel과 동일 클래스 재사용)
        self.attention = Interpreted_Multi_head_Attention(
            query_dim=hidden_dim, key_dim=hidden_dim, value_dim=hidden_dim,
            num_heads=num_heads,
        )
        self.grn_after_attention = GRN(hidden_dim, hidden_dim, hidden_dim, dropout)

        # Static feature layers
        self.static_cat_embed = (
            nn.Embedding(10, hidden_dim) if static_cat_dim is not None else None
        )
        self.static_real_layer = (
            nn.Linear(3, hidden_dim) if static_real_dim is not None else None
        )
        self.mean_cong_proj = nn.Linear(1, hidden_dim)

        # Static context를 인코더 출력에 결합하는 GRN
        self.static_grn = GRN(hidden_dim, hidden_dim, hidden_dim, dropout)

        # Reconstruction projection (deterministic)
        self.recon_proj = nn.Linear(hidden_dim, target_dim)

    def forward(self, X_past, static_cat=None, static_real=None,
                pad_mask=None, adj_matrix=None):
        B, T_past, N = X_past.shape

        # ── GCN (앞 6개 변수) ──
        GCN_idx = list(range(6))
        G = len(GCN_idx)

        hop_adj_matrix = adj_matrix[:, :6, :6]
        dist_adj_matrix = build_distribution_based_adjacency(X_past)
        dist_adj_matrix = dist_adj_matrix[:, GCN_idx, :][:, :, GCN_idx]
        adj_fused = fuse_adj_matrix(hop_adj_matrix, dist_adj_matrix, alpha=0.5)

        if pad_mask is not None:
            gcn_pad_mask = pad_mask[:, GCN_idx]
            adj_fused = adj_fused * (gcn_pad_mask.unsqueeze(1) * gcn_pad_mask.unsqueeze(2))

        x_past_gcn_input = X_past[:, :, GCN_idx].unsqueeze(-1)  # (B, T, G, 1)
        if pad_mask is not None:
            x_past_gcn_input = x_past_gcn_input * gcn_pad_mask.unsqueeze(1).unsqueeze(-1)

        B_, T_, G_, D_ = x_past_gcn_input.shape
        adj_exp = adj_fused.unsqueeze(1).repeat(1, T_, 1, 1).reshape(B_ * T_, G_, G_)
        x_reshaped = x_past_gcn_input.reshape(B_ * T_, G_, D_)
        x_past_gcn = self.graph_context(x_reshaped, adj_exp).view(B, T_past, G, -1)

        # ── Non-GCN 변수 ──
        x_past_non_gcn = X_past[:, :, G:]
        if pad_mask is not None:
            non_gcn_pad_mask = pad_mask[:, G:]
            x_past_non_gcn = x_past_non_gcn * non_gcn_pad_mask.unsqueeze(1)
        x_past_non_gcn = self.non_gcn_proj(x_past_non_gcn.unsqueeze(-1))

        # ── 결합 → VSN → LSTM Encoder ──
        x_combined = torch.cat([x_past_gcn, x_past_non_gcn], dim=2)
        x_selected, _ = self.mini_vsn(x_combined)     # (B, T_past, H)
        encoder_output, _ = self.lstm_encoder(x_selected)  # (B, T_past, H)

        # ── Temporal Self-Attention ──
        # query=key=value=encoder_output: 시계열 내 중요 구간(피크 등) 포착
        attn_output, _ = self.attention(
            encoder_output, encoder_output, encoder_output
        )
        attn_output = self.grn_after_attention(attn_output)
        encoder_output = encoder_output + attn_output  # residual connection

        # ── Static context ──
        static_context = torch.zeros(B, self.hidden_dim, device=X_past.device)
        if static_cat is not None and self.static_cat_embed is not None:
            cat_embed = self.static_cat_embed(static_cat.squeeze(-1)).mean(dim=1)
            static_context = static_context + cat_embed
        if static_real is not None and self.static_real_layer is not None:
            real_embed = self.static_real_layer(static_real[:, :, :3].mean(dim=1))
            mean_cong = static_real[:, :, 3].mean(dim=1, keepdim=True)
            real_embed = real_embed + self.mean_cong_proj(mean_cong)
            static_context = static_context + real_embed

        # Static context를 인코더 출력에 결합
        enriched = self.static_grn(encoder_output, static_context)

        # ── Reconstruction projection ──
        recon = self.recon_proj(enriched)  # (B, T_past, target_dim)
        return recon


# ============================================================================
# Reconstruction 추론 함수
# ============================================================================

_RECON_MODEL_CACHE: dict = {}
_RECON_STATS_CACHE: dict = {}  # JSON 경로 → dict

# congestion_type → reconstruction 체크포인트 파일명
_RECON_CKPT_MAP = {
    "road_congestion":     "reconstruction_pgtft_road_best.pt",
    "ped_congestion":      "reconstruction_pgtft_ped_best.pt",
    "pm_congestion":       "reconstruction_pgtft_pm_best.pt",
    "6th_road_congestion": "reconstruction_pgtft_6th_best.pt",
    "6th_ped_congestion":  "reconstruction_pgtft_6th_ped_best.pt",
}


def _load_recon_stats() -> dict:
    """recon_norm_stats.json 로드 (lazy, cache)."""
    from pathlib import Path as _Path
    import json as _json
    stats_path = _Path(__file__).parent / "model_weights" / "recon_norm_stats.json"
    key = str(stats_path)
    if key in _RECON_STATS_CACHE:
        return _RECON_STATS_CACHE[key]
    if not stats_path.exists():
        _RECON_STATS_CACHE[key] = {}
        return {}
    with stats_path.open("r", encoding="utf-8") as f:
        data = _json.load(f)
    _RECON_STATS_CACHE[key] = data
    return data


def predict_reconstruction(meta: dict) -> np.ndarray:
    """
    Reconstruction PGTFT 추론: 주변 6도로 과거 360스텝 → 타겟 도로 동일 시간대 360스텝 복원.

    학습 (train_reconstruction_pgtft_v2.py:230-233) 은 도로별 z-score 정규화를
    적용하므로, 추론에서도 동일하게 입력을 정규화하고 출력을 타겟 도로 스케일로
    역정규화해야 한다. 정규화 통계는 model_weights/recon_norm_stats.json 에서
    로드한다 (generate_recon_norm_stats.py 로 생성).

    Args:
        meta: {
            'x'                : np.ndarray (6, 360) — 이웃 도로 시계열 (raw [0,1])
            'is_padding'       : List[bool] len=6
            'neighbor_road_ids': List[str] len=6
            'target_road_id'   : str
            'congestion_type'  : str (예: 'ped_congestion')
            'timestamp'        : str
            'input_window'     : int (기본 360)
        }

    Returns:
        np.ndarray (360,) — 타겟 도로 복원 시계열 (raw 스케일, 0~1 clipped)
    """
    from pathlib import Path as _Path
    try:
        from phase_6_config import INPUT_WINDOW
    except ImportError:
        INPUT_WINDOW = 360

    congestion_type = meta.get('congestion_type', '6th_road_congestion')
    ckpt_name = _RECON_CKPT_MAP.get(congestion_type, _RECON_CKPT_MAP['6th_road_congestion'])
    _recon_ckpt = str(_Path(__file__).parent / "model_weights" / ckpt_name)

    x_raw = np.asarray(meta['x'], dtype=np.float32)       # (6, T)
    is_padding = list(meta.get('is_padding', [False] * 6))
    neighbor_ids = list(meta.get('neighbor_road_ids', [''] * 6))
    target_road_id = meta.get('target_road_id', '')
    timestamp = meta.get('timestamp', '')
    T_in = int(meta.get('input_window', INPUT_WINDOW))

    # ── 학습과 동일한 per-road z-score 정규화 ──
    stats_all = _load_recon_stats()
    ctype_stats = stats_all.get(congestion_type, {})

    x_norm = x_raw.copy()
    for i in range(min(6, x_norm.shape[0])):
        if is_padding[i]:
            continue
        rid = neighbor_ids[i] if i < len(neighbor_ids) else ''
        s = ctype_stats.get(rid)
        if s is None:
            continue
        mean = float(s['mean'])
        std = float(s['std']) if float(s['std']) != 0.0 else 1.0
        x_norm[i] = (x_norm[i] - mean) / std

    x_raw = x_norm.T  # (T_in, 6) — 정규화된 시계열

    # ── 피처 구성 (CCTVGraphDataset.prepare_sample과 동일) ──
    delta = np.diff(x_raw, axis=0, prepend=x_raw[0:1, :])
    past_ts = _build_timestamps(timestamp, T_in, future=False)
    is_peak_past = _peak_mask(past_ts).reshape(-1, 1)

    valid_mask = np.array([not p for p in is_padding[:6]], dtype=np.float32)
    n_valid = valid_mask.sum()
    one_hop_avg = (
        (x_raw * valid_mask).sum(axis=1, keepdims=True) / n_valid
        if n_valid > 0
        else np.zeros((T_in, 1), dtype=np.float32)
    )

    x_past = np.concatenate([x_raw, is_peak_past, delta, one_hop_avg], axis=1)
    # x_past: (T_in, 14)

    # ── static 피처 ──
    one_hop_mean_static = float(one_hop_avg.mean())
    static_cat = np.zeros((6, 1), dtype=np.float32)
    static_real = np.array(
        [[float(valid_mask[i]), 0.0, 0.0, one_hop_mean_static] for i in range(6)],
        dtype=np.float32,
    )

    # ── pad_mask (14,) ──
    one_hop_valid = bool(one_hop_avg.any())
    pad_mask = np.concatenate([
        valid_mask,
        [1.0],
        valid_mask,
        [1.0 if one_hop_valid else 0.0],
    ]).astype(np.float32)

    # ── adj_matrix (6, 6) ──
    adj = np.zeros((6, 6), dtype=np.float32)
    for i in range(6):
        if is_padding[i]:
            continue
        for j in range(6):
            if not is_padding[j]:
                adj[i, j] = 1.0
        adj[i, i] = 1.0

    # ── 마스크 적용 ──
    pad_exp = pad_mask[np.newaxis, np.newaxis, :]
    x_past_b = x_past[np.newaxis] * pad_exp
    static_real = static_real * pad_mask[:6, np.newaxis]
    static_cat = (static_cat * pad_mask[:6, np.newaxis]).astype(np.int64)

    # ── 텐서 변환 & 추론 ──
    device = _get_device()
    def _t(a, dt): return torch.tensor(a, dtype=dt).to(device)

    with torch.no_grad():
        pred = _load_recon_model(_recon_ckpt, device)(
            _t(x_past_b, torch.float32),
            static_cat=_t(static_cat[np.newaxis], torch.int64),
            static_real=_t(static_real[np.newaxis], torch.float32),
            pad_mask=_t(pad_mask[np.newaxis], torch.float32),
            adj_matrix=_t(adj[np.newaxis], torch.float32),
        )
    # pred: (1, T_in, 1) → squeeze (z-score 스케일)
    recon_series = pred[0, :, 0].cpu().numpy()

    # ── 타겟 도로 스케일로 역정규화 ──
    tgt_stats = ctype_stats.get(target_road_id)
    if tgt_stats is not None:
        t_mean = float(tgt_stats['mean'])
        t_std = float(tgt_stats['std']) if float(tgt_stats['std']) != 0.0 else 1.0
        recon_series = recon_series * t_std + t_mean

    return np.clip(recon_series, 0.0, 1.0).astype(np.float32)


def _load_recon_model(checkpoint_path: str, device, num_inputs: int = 14):
    """Reconstruction 체크포인트에서 ReconstructionTFTModel 로드 (캐시)."""
    from pathlib import Path as _P
    cache_key = (checkpoint_path, str(device))
    if cache_key in _RECON_MODEL_CACHE:
        return _RECON_MODEL_CACHE[cache_key]

    model = ReconstructionTFTModel(
        input_dim=num_inputs,
        hidden_dim=256,
        num_heads=8,
        num_layers=2,
        target_dim=1,
        dropout=0.1,
        num_inputs=num_inputs,
        static_cat_dim=1,
        static_real_dim=4,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    _RECON_MODEL_CACHE[cache_key] = model
    return model


# ============================================================================
# 추론 진입점 — phase_3_path_construction._run_pgtft() 에서 호출
# ============================================================================

_PGTFT_MODEL_CACHE: dict = {}   # (checkpoint_path, device_str) → TFTModel


def _calendar_inference_features(timestamps) -> np.ndarray:
    """(T, 2) TFT-paper canonical [hour/23, dow/6]. Mirrors preprocessing._calendar_features."""
    hour = np.asarray(timestamps.hour, dtype=np.float32) / 23.0
    dow = np.asarray(timestamps.dayofweek, dtype=np.float32) / 6.0
    return np.stack([hour, dow], axis=1).astype(np.float32)


def _apply_inference_calendar_mode(
    x_past_base: np.ndarray,
    x_future_base: np.ndarray,
    pad_mask_base: np.ndarray,
    past_ts,
    fut_ts,
    calendar_future_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror of preprocessing.apply_calendar_mode_to_past / _build_inference_style_future
    for inference path. Inputs are BASE 14-dim layouts.

    Returns (x_past, x_future, pad_mask) in mode-specific layout (14/15/16-dim).
    """
    if calendar_future_mode == "off":
        return (
            x_past_base.astype(np.float32),
            x_future_base.astype(np.float32),
            pad_mask_base.astype(np.float32),
        )

    cal_past = _calendar_inference_features(past_ts)
    cal_fut = _calendar_inference_features(fut_ts)

    if calendar_future_mode == "swap":
        x_past = np.concatenate(
            [x_past_base[:, :6], cal_past, x_past_base[:, 7:]], axis=1
        )
        x_future = np.concatenate(
            [x_future_base[:, :6], cal_fut, x_future_base[:, 7:]], axis=1
        )
        pad_mask = np.concatenate(
            [pad_mask_base[:6], [1.0, 1.0], pad_mask_base[7:]]
        )
    elif calendar_future_mode == "add":
        x_past = np.concatenate([x_past_base, cal_past], axis=1)
        x_future = np.concatenate([x_future_base, cal_fut], axis=1)
        pad_mask = np.concatenate([pad_mask_base, [1.0, 1.0]])
    else:
        raise ValueError(f"unknown calendar_future_mode={calendar_future_mode!r}")

    return x_past.astype(np.float32), x_future.astype(np.float32), pad_mask.astype(np.float32)


# ============================================================================
# 학습 포맷 static_real / adj_matrix 구성 (fixed-format 패치)
# ============================================================================
#
# 배경:
#   이전 추론 코드는 static_real = [valid, 0, 0, one_hop_mean], adj = binary
#   로 PGTFT 에 들어갔지만, 학습 시엔
#     static_real = [hop_weight=1.0, length_norm, width_norm, one_hop_mean]
#     adj[i,j]    = 1 / (1 + |hop_i - hop_j|)
#   였음. Format mismatch 로 운영 데이터에서 출력이 입력 변동에 둔감해지고
#   (swap/LOO first std ~ 0.0004 수준), 평균도 GT 대비 편향이 컸음.
#   diagnose_fixed_vs_broken.py 진단으로 fixed format 이 swap 민감도를
#   13~44× 회복함을 확인 → 영구 패치.
#
# fallback 동작:
#   meta 에 length_norms/width_norms/hops 가 없으면 학습 분포 중앙값(0.5) 와
#   hop 근사 [0,1,1,1,2,2,2] 를 쓴다. 현재 pipeline 의 prepare_sequence_metadata
#   가 이 키들을 채워주므로 fallback 은 외부 호출자용 안전장치.


def _build_static_real_and_adj(
    is_padding,
    one_hop_mean_static: float,
    length_norms=None,
    width_norms=None,
    hops=None,
):
    """학습 포맷 static_real(6,4) + adj(6,6) 반환.

    Args:
        is_padding        : List[bool] len=6
        one_hop_mean_static: float — valid 도로 평균 혼잡도 (static_real[:,3])
        length_norms, width_norms: Optional[List[float]] len=6 — 각 도로 LENGTH/WIDTH
                                   min-max 정규화 값. None 이면 0.5 fallback.
        hops              : Optional[List[int]] len=6 — 각 도로의 target 기준 hop.
                            None 이면 [0,1,1,1,2,2,2] 중 앞 6 개 fallback.

    Returns:
        (static_real, adj) — np.ndarray, np.ndarray
    """
    # --- length / width 정규화 값 ---
    if length_norms is None:
        length_norms = [0.5] * 6
    else:
        length_norms = list(length_norms) + [0.5] * max(0, 6 - len(length_norms))
        length_norms = length_norms[:6]
    if width_norms is None:
        width_norms = [0.5] * 6
    else:
        width_norms = list(width_norms) + [0.5] * max(0, 6 - len(width_norms))
        width_norms = width_norms[:6]

    # --- hop (adj similarity 용) ---
    if hops is None:
        hops = [0, 1, 1, 1, 2, 2]
    else:
        # 슬롯 0 은 target 이라 0 이 자연스럽지만 호출자가 넘긴 값을 우선
        hops = list(hops) + [2] * max(0, 6 - len(hops))
        hops = hops[:6]

    # --- static_real: [hop_weight=1.0 if valid else 0.0, len_n, wid_n, one_hop_mean] ---
    static_real = np.zeros((6, 4), dtype=np.float32)
    for i in range(6):
        hop_weight = 0.0 if is_padding[i] else 1.0
        static_real[i, 0] = hop_weight
        static_real[i, 1] = float(length_norms[i])
        static_real[i, 2] = float(width_norms[i])
        static_real[i, 3] = float(one_hop_mean_static)

    # --- adj[i,j] = 1/(1+|h_i - h_j|)  (둘 다 valid 일 때만) ---
    adj = np.zeros((6, 6), dtype=np.float32)
    for i in range(6):
        if is_padding[i]:
            continue
        for j in range(6):
            if is_padding[j]:
                continue
            adj[i, j] = 1.0 / (1.0 + abs(int(hops[i]) - int(hops[j])))
    return static_real, adj


def _build_static_cat(is_padding, hops=None):
    """Build discrete hop category for static embedding.

    PAD nodes use 0. Real neighbors use integer hop categories 1..9, matching
    the `nn.Embedding(10, hidden_dim)` range.
    """
    if hops is None:
        hops = [1, 1, 1, 2, 2, 2]
    else:
        hops = list(hops) + [2] * max(0, 6 - len(hops))
        hops = hops[:6]

    static_cat = np.zeros((6, 1), dtype=np.int64)
    for i in range(6):
        if is_padding[i]:
            static_cat[i, 0] = 0
        else:
            static_cat[i, 0] = int(np.clip(int(hops[i]), 1, 9))
    return static_cat


def predict(meta: dict) -> float:
    """
    TFTModel(= peak_awared_test)로 음영 도로 혼잡도 예측.

    phase_3_path_construction._run_pgtft()에서 호출.

    Args:
        meta: phase_2_sequence_generation.prepare_sequence_metadata() 반환값
              {
                'x'               : np.ndarray (6, input_window)
                'is_padding'      : List[bool]  len=6
                'timestamp'       : str   ("2024-03-20 14:00" 형식)
                'input_window'    : int   (기본 360)
                'forecast_horizon': int   (기본 120)
              }

    Returns:
        Q50 예측 혼잡도 float (0.0 ~ 1.0 clipped)
    """
    # ── 설정 로드 ──────────────────────────────────────────────────────────
    try:
        from phase_6_config import (
            INPUT_WINDOW, FORECAST_HORIZON, FORECAST_TARGET_STEP,
            PGTFT_CHECKPOINT_PATH,
        )
    except ImportError:
        from pathlib import Path
        INPUT_WINDOW           = 360
        FORECAST_HORIZON       = 120
        FORECAST_TARGET_STEP   = 0
        PGTFT_CHECKPOINT_PATH  = str(
            Path(__file__).parent / "model_weights" /
            "forecasting_pgtft_6th_ped_best.pt"
        )

    # ── meta 파싱 ──────────────────────────────────────────────────────────
    x_raw      = np.asarray(meta['x'], dtype=np.float32)           # (6, T)
    is_padding = list(meta.get('is_padding', [False] * 6))
    timestamp  = meta.get('timestamp', '')
    T_in       = int(meta.get('input_window',    INPUT_WINDOW))
    T_fut      = int(meta.get('forecast_horizon', FORECAST_HORIZON))
    # 학습 포맷 static feature (prepare_sequence_metadata 가 채워줌)
    length_norms = meta.get('length_norms')
    width_norms  = meta.get('width_norms')
    hops         = meta.get('hops')

    x_raw = x_raw.T                                                 # (T_in, 6)

    # ── 피처 구성 (CCTVGraphDataset.prepare_sample 과 동일한 순서) ───────
    #   x_past:   [raw(6) | peak_hour(1) | delta(6) | one_hop_avg(1)] = 14 ch
    delta        = np.diff(x_raw, axis=0, prepend=x_raw[0:1, :])   # (T_in, 6)
    past_ts      = _build_timestamps(timestamp, T_in, future=False)
    is_peak_past = _peak_mask(past_ts).reshape(-1, 1)               # (T_in, 1)

    valid_mask = np.array([not p for p in is_padding[:6]], dtype=np.float32)
    n_valid    = valid_mask.sum()
    one_hop_avg = (
        (x_raw * valid_mask).sum(axis=1, keepdims=True) / n_valid
        if n_valid > 0
        else np.zeros((T_in, 1), dtype=np.float32)
    )                                                               # (T_in, 1)

    x_past_base = np.concatenate([x_raw, is_peak_past, delta, one_hop_avg], axis=1)
    # x_past_base: (T_in, 14)

    # ── x_future: 마지막 관측값 고정 + peak_hour 재계산 ──────────────────
    fut_ts      = _build_timestamps(timestamp, T_fut, future=True)
    is_peak_fut = _peak_mask(fut_ts).reshape(-1, 1)                 # (T_fut, 1)
    fut_raw     = np.tile(x_raw[-1:, :],       (T_fut, 1))          # (T_fut, 6)
    fut_delta   = np.zeros((T_fut, 6), dtype=np.float32)
    fut_one_hop = np.tile(one_hop_avg[-1:, :], (T_fut, 1))          # (T_fut, 1)
    x_future_base = np.concatenate([fut_raw, is_peak_fut, fut_delta, fut_one_hop], axis=1)
    # x_future_base: (T_fut, 14)

    # ── static 피처 (학습 포맷: [hop_weight, len_n, wid_n, one_hop_mean]) ─
    one_hop_mean_static = float(one_hop_avg.mean())
    static_cat  = _build_static_cat(is_padding[:6], hops=hops)
    static_real, adj = _build_static_real_and_adj(
        is_padding=is_padding[:6],
        one_hop_mean_static=one_hop_mean_static,
        length_norms=length_norms,
        width_norms=width_norms,
        hops=hops,
    )                                                               # (6,4), (6,6)

    # ── pad_mask (14,) — TFTSlidingWindowDataset 와 동일한 구성 ───────────
    one_hop_valid = bool(one_hop_avg.any())
    pad_mask_base = np.concatenate([
        valid_mask,                                  # (6,) — raw 채널
        [1.0],                                       # peak_hour 항상 유효
        valid_mask,                                  # (6,) — delta 채널
        [1.0 if one_hop_valid else 0.0],             # one_hop_avg
    ]).astype(np.float32)                           # (14,)

    # ── R1-A: calendar_future_mode (off|swap|add) ─────────────────────────
    device = _get_device()
    model = _load_pgtft_model(PGTFT_CHECKPOINT_PATH, device)
    mode = getattr(model, "calendar_future_mode", "off")
    x_past, x_future, pad_mask = _apply_inference_calendar_mode(
        x_past_base, x_future_base, pad_mask_base, past_ts, fut_ts, mode,
    )

    # ── 마스크 적용 (Best_Model_Test 와 동일) ─────────────────────────────
    pad_exp     = pad_mask[np.newaxis, np.newaxis, :]               # (1, 1, N)
    x_past_b    = x_past[np.newaxis]   * pad_exp                    # (1, T_in,  N)
    x_future_b  = x_future[np.newaxis] * pad_exp                    # (1, T_fut, N)
    static_real = static_real * pad_mask_base[:6, np.newaxis]       # (6, 4) — base 14-dim 의 raw valid 사용
    static_cat  = (static_cat * pad_mask_base[:6, np.newaxis]).astype(np.int64)  # (6, 1)

    # ── 텐서 변환 & 추론 ──────────────────────────────────────────────────
    def _t(a, dt): return torch.tensor(a, dtype=dt).to(device)

    with torch.no_grad():
        pred = model(
            _t(x_past_b,               torch.float32),
            _t(x_future_b,             torch.float32),
            _t(static_cat[np.newaxis], torch.int64),
            _t(static_real[np.newaxis],torch.float32),
            pad_mask  = _t(pad_mask[np.newaxis],   torch.float32),
            adj_matrix= _t(adj[np.newaxis],        torch.float32),
        )
    # pred: (1, T_fut, 1, 3) → Q50 = index 1
    q50_series = pred[0, :, 0, 1].cpu().numpy()               # (T_fut,)
    q50_series = np.clip(q50_series, 0.0, 1.0)
    return float(q50_series[FORECAST_TARGET_STEP])


def _predict_raw(meta: dict) -> np.ndarray:
    """
    predict()와 동일한 전처리 후 forecast_horizon 전체를 반환.

    Returns:
        np.ndarray shape (forecast_horizon,) — Q50 예측값 0~1 clipped
    """
    try:
        from phase_6_config import (
            INPUT_WINDOW, FORECAST_HORIZON,
            PGTFT_CHECKPOINT_PATH,
        )
    except ImportError:
        from pathlib import Path
        INPUT_WINDOW           = 360
        FORECAST_HORIZON       = 120
        PGTFT_CHECKPOINT_PATH  = str(
            Path(__file__).parent / "model_weights" /
            "forecasting_pgtft_6th_ped_best.pt"
        )

    x_raw      = np.asarray(meta['x'], dtype=np.float32)           # (6, T)
    is_padding = list(meta.get('is_padding', [False] * 6))
    timestamp  = meta.get('timestamp', '')
    T_in       = int(meta.get('input_window',    INPUT_WINDOW))
    T_fut      = int(meta.get('forecast_horizon', FORECAST_HORIZON))
    length_norms = meta.get('length_norms')
    width_norms  = meta.get('width_norms')
    hops         = meta.get('hops')

    x_raw = x_raw.T                                                 # (T_in, 6)

    delta        = np.diff(x_raw, axis=0, prepend=x_raw[0:1, :])
    past_ts      = _build_timestamps(timestamp, T_in, future=False)
    is_peak_past = _peak_mask(past_ts).reshape(-1, 1)

    valid_mask = np.array([not p for p in is_padding[:6]], dtype=np.float32)
    n_valid    = valid_mask.sum()
    one_hop_avg = (
        (x_raw * valid_mask).sum(axis=1, keepdims=True) / n_valid
        if n_valid > 0
        else np.zeros((T_in, 1), dtype=np.float32)
    )

    x_past_base = np.concatenate([x_raw, is_peak_past, delta, one_hop_avg], axis=1)

    fut_ts      = _build_timestamps(timestamp, T_fut, future=True)
    is_peak_fut = _peak_mask(fut_ts).reshape(-1, 1)
    fut_raw     = np.tile(x_raw[-1:, :],       (T_fut, 1))
    fut_delta   = np.zeros((T_fut, 6), dtype=np.float32)
    fut_one_hop = np.tile(one_hop_avg[-1:, :], (T_fut, 1))
    x_future_base = np.concatenate([fut_raw, is_peak_fut, fut_delta, fut_one_hop], axis=1)

    one_hop_mean_static = float(one_hop_avg.mean())
    static_cat  = _build_static_cat(is_padding[:6], hops=hops)
    static_real, adj = _build_static_real_and_adj(
        is_padding=is_padding[:6],
        one_hop_mean_static=one_hop_mean_static,
        length_norms=length_norms,
        width_norms=width_norms,
        hops=hops,
    )

    one_hop_valid = bool(one_hop_avg.any())
    pad_mask_base = np.concatenate([
        valid_mask,
        [1.0],
        valid_mask,
        [1.0 if one_hop_valid else 0.0],
    ]).astype(np.float32)

    device = _get_device()
    model = _load_pgtft_model(PGTFT_CHECKPOINT_PATH, device)
    mode = getattr(model, "calendar_future_mode", "off")
    x_past, x_future, pad_mask = _apply_inference_calendar_mode(
        x_past_base, x_future_base, pad_mask_base, past_ts, fut_ts, mode,
    )

    pad_exp     = pad_mask[np.newaxis, np.newaxis, :]
    x_past_b    = x_past[np.newaxis]   * pad_exp
    x_future_b  = x_future[np.newaxis] * pad_exp
    static_real = static_real * pad_mask_base[:6, np.newaxis]
    static_cat  = (static_cat * pad_mask_base[:6, np.newaxis]).astype(np.int64)

    def _t(a, dt): return torch.tensor(a, dtype=dt).to(device)

    with torch.no_grad():
        pred = model(
            _t(x_past_b,               torch.float32),
            _t(x_future_b,             torch.float32),
            _t(static_cat[np.newaxis], torch.int64),
            _t(static_real[np.newaxis],torch.float32),
            pad_mask  = _t(pad_mask[np.newaxis],   torch.float32),
            adj_matrix= _t(adj[np.newaxis],        torch.float32),
        )
    # pred: (1, T_fut, 1, 3) → Q50 전체
    q50_series = pred[0, :, 0, 1].cpu().numpy()
    return np.clip(q50_series, 0.0, 1.0).astype(np.float32)


def predict_series(
    neighbor_series: np.ndarray,
    is_padding: list,
    timestamps,
    input_window: int = 360,
    forecast_horizon: int = 120,
) -> np.ndarray:
    """
    슬라이딩 윈도우로 PGTFT를 반복 호출하여 타겟 도로의 전체 시계열을 예측.

    Args:
        neighbor_series : np.ndarray (6, T_total) — 이웃 6개 도로의 전체 시계열
        is_padding      : List[bool] len=6 — 각 이웃의 패딩 여부
        timestamps      : pd.DatetimeIndex len=T_total — 전체 시간 인덱스
        input_window    : 과거 lookback (기본 360 = 6시간)
        forecast_horizon: 예측 범위 (기본 120 = 2시간)

    Returns:
        np.ndarray (T_total,) — 예측값. 처음 input_window 구간은 NaN.
    """
    import logging
    logger = logging.getLogger(__name__)

    T_total = neighbor_series.shape[1]
    predictions = np.full(T_total, np.nan, dtype=np.float32)

    stride = forecast_horizon
    n_windows = 0

    for start in range(0, T_total - input_window, stride):
        window_end = start + input_window
        pred_end   = min(window_end + forecast_horizon, T_total)
        pred_len   = pred_end - window_end

        if pred_len <= 0:
            break

        x_window = neighbor_series[:, start:window_end]  # (6, input_window)
        ref_ts   = str(timestamps[window_end - 1])

        meta = {
            'x': x_window,
            'is_padding': is_padding,
            'timestamp': ref_ts,
            'input_window': input_window,
            'forecast_horizon': forecast_horizon,
        }

        try:
            window_pred = _predict_raw(meta)           # (forecast_horizon,)
            predictions[window_end:pred_end] = window_pred[:pred_len]
            n_windows += 1
        except Exception as e:
            logger.warning(f"  predict_series window {n_windows} 실패: {e}")
            continue

    logger.info(f"  predict_series: {n_windows}개 윈도우 실행, "
                f"예측 커버리지 {np.count_nonzero(~np.isnan(predictions))}/{T_total}")
    return predictions


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _get_device():
    if platform.system() == "Darwin":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_pgtft_model(checkpoint_path: str, device, num_inputs: int = 14):
    """체크포인트에서 TFTModel 로드 (모듈 수준 캐시).

    두 포맷 자동 감지:
    - 신규 (`train_forecasting_pgtft.py` 산출): dict
        {"model_state_dict": ..., "epoch": ..., "val_loss": ..., "config": {...}}
        → config의 hidden_dim/num_heads/num_layers/dropout/num_quantiles 사용
    - 레거시 (`density_estimation/PGTFT/...`): raw state_dict
        → num_heads=8 등 옛 학습 시 디폴트로 모델 구성

    Round 1 R1-A: cfg["calendar_future_mode"] ∈ {off, swap, add}
      → num_inputs adjusts to 14/15/16 and is recorded as `model.calendar_future_mode`
        so _predict_raw/predict can rebuild matching x_past/x_future/pad_mask.
    """
    cache_key = (checkpoint_path, str(device))
    if cache_key in _PGTFT_MODEL_CACHE:
        return _PGTFT_MODEL_CACHE[cache_key]

    obj = torch.load(checkpoint_path, map_location=device)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
        cfg = obj.get("config", {}) or {}
        hidden_dim    = int(cfg.get("hidden_dim", 128))
        num_heads     = int(cfg.get("num_heads", 4))
        num_layers    = int(cfg.get("num_layers", 2))
        dropout       = float(cfg.get("dropout", 0.1))
        num_quantiles = int(cfg.get("num_quantiles", 3))
        use_adaptive_adj = bool(cfg.get("use_adaptive_adj", False))
        node_embed_dim   = int(cfg.get("node_embed_dim", 10))
        use_napl         = bool(cfg.get("use_napl", False))
        use_mixstyle     = bool(cfg.get("use_mixstyle", False))
        mixstyle_p       = float(cfg.get("mixstyle_p", 0.5))
        mixstyle_alpha   = float(cfg.get("mixstyle_alpha", 0.1))
        use_anchor_residual = bool(cfg.get("use_anchor_residual", False))
        anchor_topk         = int(cfg.get("anchor_topk", 6))
        _ac = cfg.get("anchor_clamp", None)
        anchor_clamp        = float(_ac) if _ac is not None else None
        use_anchor_gate     = bool(cfg.get("use_anchor_gate", False))
        anchor_gate_bias_init = float(cfg.get("anchor_gate_bias_init", 0.0))
        use_horizon_skip    = bool(cfg.get("use_horizon_skip", False))
        hskip_w_max         = float(cfg.get("hskip_w_max", 0.3))
        hskip_tau           = float(cfg.get("hskip_tau", 60.0))
        calendar_future_mode = str(cfg.get("calendar_future_mode", "off"))
        # L6a (R2-A): default True 로 두면 기존 ckpt (키 없음) 도 영향 없음.
        use_peak_grn        = bool(cfg.get("use_peak_grn", True))
        use_dist_adj        = bool(cfg.get("use_dist_adj", True))
        # L6b (R2-B): default True = 기존 PGTFT 동작 그대로.
        use_vsn             = bool(cfg.get("use_vsn", True))
        use_attn_grn        = bool(cfg.get("use_attn_grn", True))
        use_final_gate_grn  = bool(cfg.get("use_final_gate_grn", True))
        # R2-C1: default = 기존 PGTFT 동작 (SoftAttentionGCN on, TCN off).
        use_soft_gcn        = bool(cfg.get("use_soft_gcn", True))
        use_tcn             = bool(cfg.get("use_tcn", False))
        tcn_num_layers      = int(cfg.get("tcn_num_layers", 4))
        tcn_kernel_size     = int(cfg.get("tcn_kernel_size", 2))
        _td = cfg.get("tcn_dilations", None)
        tcn_dilations       = [int(d) for d in _td] if _td else None
        tcn_activation      = str(cfg.get("tcn_activation", "relu"))
        tcn_use_peak_gate   = bool(cfg.get("tcn_use_peak_gate", False))
        tcn_peak_alpha      = float(cfg.get("tcn_peak_alpha", 0.3))
        use_target_node_select = bool(cfg.get("use_target_node_select", False))
        # R3-LSTGF: default False / 2 / 6 / 0.0 (no-op).
        use_lstgf           = bool(cfg.get("use_lstgf", False))
        lstgf_K             = int(cfg.get("lstgf_K", 2))
        lstgf_R             = int(cfg.get("lstgf_R", 6))
        lstgf_gamma_init    = float(cfg.get("lstgf_gamma_init", 0.0))
        # Track A 4-cell (2026-05-14): strip_aug / strip_static 우선, legacy strip_features alias.
        _legacy_strip       = bool(cfg.get("strip_features", False))
        _sa_cfg             = cfg.get("strip_aug", None)
        _ss_cfg             = cfg.get("strip_static", None)
        strip_aug           = bool(_sa_cfg) if _sa_cfg is not None else _legacy_strip
        strip_static        = bool(_ss_cfg) if _ss_cfg is not None else _legacy_strip
        ckpt_num_inputs     = cfg.get("num_inputs", None)
    else:
        state_dict = obj
        hidden_dim    = 128
        num_heads     = 8
        num_layers    = 2
        dropout       = 0.1
        num_quantiles = 3
        use_adaptive_adj = False
        node_embed_dim   = 10
        use_napl         = False
        use_mixstyle     = False
        mixstyle_p       = 0.5
        mixstyle_alpha   = 0.1
        use_anchor_residual = False
        anchor_topk         = 6
        anchor_clamp        = None
        use_anchor_gate     = False
        anchor_gate_bias_init = 0.0
        use_horizon_skip    = False
        hskip_w_max         = 0.3
        hskip_tau           = 60.0
        calendar_future_mode = "off"
        use_peak_grn        = True
        use_dist_adj        = True
        use_vsn             = True
        use_attn_grn        = True
        use_final_gate_grn  = True
        use_soft_gcn        = True
        use_tcn             = False
        tcn_num_layers      = 4
        tcn_kernel_size     = 2
        tcn_dilations       = None
        tcn_activation      = "relu"
        tcn_use_peak_gate   = False
        tcn_peak_alpha      = 0.3
        use_target_node_select = False
        use_lstgf           = False
        lstgf_K             = 2
        lstgf_R             = 6
        lstgf_gamma_init    = 0.0
        strip_aug           = False
        strip_static        = False
        ckpt_num_inputs     = None

    # Resolve num_inputs: prefer ckpt's recorded num_inputs (covers calendar modes 14/15/16).
    if ckpt_num_inputs is not None:
        num_inputs = int(ckpt_num_inputs)
    elif calendar_future_mode == "swap":
        num_inputs = 15
    elif calendar_future_mode == "add":
        num_inputs = 16
    # else: caller default (14)

    model = TFTModel(
        input_dim       = num_inputs,
        hidden_dim      = hidden_dim,
        num_heads       = num_heads,
        num_layers      = num_layers,
        target_dim      = 1,
        dropout         = dropout,
        num_quantiles   = num_quantiles,
        num_inputs      = num_inputs,
        static_cat_dim  = 1,
        static_real_dim = 4,
        use_adaptive_adj = use_adaptive_adj,
        node_embed_dim   = node_embed_dim,
        use_napl         = use_napl,
        use_mixstyle     = use_mixstyle,
        mixstyle_p       = mixstyle_p,
        mixstyle_alpha   = mixstyle_alpha,
        use_anchor_residual = use_anchor_residual,
        anchor_topk         = anchor_topk,
        anchor_clamp        = anchor_clamp,
        use_anchor_gate         = use_anchor_gate,
        anchor_gate_bias_init   = anchor_gate_bias_init,
        use_horizon_skip        = use_horizon_skip,
        hskip_w_max             = hskip_w_max,
        hskip_tau               = hskip_tau,
        use_peak_grn            = use_peak_grn,
        use_dist_adj            = use_dist_adj,
        use_vsn                 = use_vsn,
        use_attn_grn            = use_attn_grn,
        use_final_gate_grn      = use_final_gate_grn,
        use_soft_gcn            = use_soft_gcn,
        use_tcn                 = use_tcn,
        tcn_num_layers          = tcn_num_layers,
        tcn_kernel_size         = tcn_kernel_size,
        tcn_dilations           = tcn_dilations,
        tcn_activation          = tcn_activation,
        tcn_use_peak_gate       = tcn_use_peak_gate,
        tcn_peak_alpha          = tcn_peak_alpha,
        use_target_node_select  = use_target_node_select,
        use_lstgf               = use_lstgf,
        lstgf_K                 = lstgf_K,
        lstgf_R                 = lstgf_R,
        lstgf_gamma_init        = lstgf_gamma_init,
        strip_aug               = strip_aug,
        strip_static            = strip_static,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    # Stash for _predict_raw/predict to reconstruct matching inputs.
    model.calendar_future_mode = calendar_future_mode
    _PGTFT_MODEL_CACHE[cache_key] = model
    return model


def _build_timestamps(ref_ts_str: str, length: int, future: bool = False):
    """ref_ts_str 기준 length개 분(minute) 단위 DatetimeIndex 생성."""
    import pandas as pd
    try:
        ref = pd.Timestamp(ref_ts_str)
    except Exception:
        ref = pd.Timestamp.now()
    if future:
        return pd.date_range(start=ref, periods=length, freq="1min")
    return pd.date_range(end=ref, periods=length, freq="1min")


def _peak_mask(timestamps) -> np.ndarray:
    """피크 시간대 여부 float32 배열 반환 (7-10, 12-14, 17-20)."""
    h = timestamps.hour
    return (
        ((h >= 7)  & (h < 10)) |
        ((h >= 12) & (h < 14)) |
        ((h >= 17) & (h < 20))
    ).astype(np.float32)
