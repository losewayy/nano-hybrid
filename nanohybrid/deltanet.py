"""Gated DeltaNet layer (Qwen3-Next / Qwen3.8-Next style).

Recurrence per head (state S in R^{dk x dv}):
    S_t = (exp(g_t) * S_{t-1}) (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    o_t = S_t^T q_t        (q pre-scaled by 1/sqrt(dk))

Two compute modes, both pure PyTorch:
  - delta_rule_chunk: chunked form for training (matmul-friendly)
  - delta_rule_recurrent: step-by-step for cached decoding

Reference: fla-org/flash-linear-attention naive implementations +
transformers Qwen3NextGatedDeltaNet wiring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RMSNormGated


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def delta_rule_recurrent(q, k, v, beta, g, state=None):
    """Step-by-step gated delta rule.

    q,k,v: [B, H, T, D];  beta, g: [B, H, T];  state: [B, H, K, V] or None.
    Returns (o [B,H,T,V], final_state).
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    q = q.float() * (K ** -0.5)
    k, v, beta, g = k.float(), v.float(), beta.float(), g.float()
    h = q.new_zeros(B, H, K, V) if state is None else state.float()
    outs = []
    for i in range(T):
        h = h * g[:, :, i].exp()[..., None, None]
        b_v = v[:, :, i] - (h * k[:, :, i][..., None]).sum(-2)
        b_v = b_v * beta[:, :, i][..., None]
        h = h + k[:, :, i].unsqueeze(-1) * b_v.unsqueeze(-2)
        outs.append(torch.einsum("bhd,bhdm->bhm", q[:, :, i], h))
    return torch.stack(outs, dim=2), h


def delta_rule_chunk(q, k, v, beta, g, state=None, chunk_size=64):
    """Chunked gated delta rule (training path), ported from HF torch_chunk_gated_delta_rule.

    q,k,v: [B, H, T, D];  beta, g: [B, H, T] (g = log-decay <= 0).
    Returns (o [B,H,T,V], final_state).
    """
    B, H, T, K = q.shape
    q, k, v = (x.float() for x in (q, k, v))
    beta, decay = beta.float(), g.float()
    q = q * (K ** -0.5)

    BT = chunk_size
    pad = (BT - T % BT) % BT
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))
    n_chunks = q.shape[2] // BT

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, k_beta, v_beta = (x.reshape(B, H, -1, BT, x.shape[-1])
                            for x in (q, k, k_beta, v_beta))
    decay = decay.reshape(B, H, -1, BT)

    upper = torch.ones(BT, BT, dtype=torch.bool, device=q.device).triu(1)
    cum = decay.cumsum(-1)
    pairwise = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()

    ut_system = (k_beta @ k.transpose(-1, -2)) * pairwise
    intra_attn = (q @ k.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum.exp().unsqueeze(-1)

    # (I + strictly-lower(ut_system)) x = rhs — unitriangular solve, autograd-safe
    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)

    q = q * cum.exp().unsqueeze(-1)
    k = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]

    S = q.new_zeros(B, H, K, v.shape[-1]) if state is None else state.float()
    o = torch.zeros_like(new_values)
    for i in range(n_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ S
        o[:, :, i] = q[:, :, i] @ S + intra_attn[:, :, i] @ v_new
        S = S * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ v_new

    return o.reshape(B, H, -1, v.shape[-1])[:, :, :T], S


class GatedDeltaNet(nn.Module):
    """Qwen3-Next GDN layer, simplified: num_k_heads == num_v_heads."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.H = cfg.gdn_heads
        self.dk = cfg.gdn_head_k
        self.dv = cfg.gdn_head_v
        key_dim, val_dim = self.H * self.dk, self.H * self.dv
        self.conv_dim = 2 * key_dim + val_dim

        self.in_proj_qkvz = nn.Linear(d, 2 * key_dim + 2 * val_dim, bias=False)
        self.in_proj_ba = nn.Linear(d, 2 * self.H, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, kernel_size=cfg.gdn_conv_kernel,
            groups=self.conv_dim, padding=cfg.gdn_conv_kernel - 1, bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.H))
        A = torch.empty(self.H).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = RMSNormGated(self.dv)
        self.out_proj = nn.Linear(val_dim, d, bias=False)

    def _project(self, x):
        B, T, _ = x.shape
        qkvz = self.in_proj_qkvz(x)
        q, k, v, z = qkvz.split(
            [self.H * self.dk, self.H * self.dk, self.H * self.dv, self.H * self.dv], dim=-1
        )
        ba = self.in_proj_ba(x)
        b, a = ba.split(self.H, dim=-1)
        return q, k, v, z, b, a

    def forward(self, x, cache=None):
        """Training/prefill path. cache dict keys 'conv'/'rec' for decode mode."""
        B, T, _ = x.shape
        q, k, v, z, b, a = self._project(x)

        mixed_in = torch.cat([q, k, v], dim=-1).transpose(1, 2)       # [B, conv_dim, T]
        K = self.conv1d.kernel_size[0]
        history = cache.get("conv") if cache is not None else None    # [B, conv_dim, K-1]
        if history is not None:
            mixed_in = torch.cat([history, mixed_in], dim=2)
            mixed = F.conv1d(mixed_in, self.conv1d.weight, self.conv1d.bias,
                             padding=0, groups=self.conv_dim)
        else:
            mixed = F.conv1d(mixed_in, self.conv1d.weight, self.conv1d.bias,
                             padding=K - 1, groups=self.conv_dim)[..., :T]
        if cache is not None:
            if mixed_in.shape[2] >= K - 1:
                cache["conv"] = mixed_in[..., -(K - 1):].detach()
            else:
                cache["conv"] = F.pad(mixed_in.detach(), (K - 1 - mixed_in.shape[2], 0))
        mixed = F.silu(mixed).transpose(1, 2)

        ks, vs = self.H * self.dk, self.H * self.dv
        q, k, v = mixed.split([ks, ks, vs], dim=-1)
        q = q.view(B, T, self.H, self.dk).transpose(1, 2)
        k = k.view(B, T, self.H, self.dk).transpose(1, 2)
        v = v.view(B, T, self.H, self.dv).transpose(1, 2)

        # l2norm in fp32, matching HF/fla kernels (norm done inside kernel at fp32)
        q, k = l2norm(q.float()), l2norm(k.float())
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        beta, g = beta.transpose(1, 2), g.transpose(1, 2)             # [B,H,T]
        init = cache.get("rec") if cache is not None else None
        if T == 1:
            o, S = delta_rule_recurrent(q, k, v, beta, g, state=init)
        else:
            o, S = delta_rule_chunk(q, k, v, beta, g, state=init)
        if cache is not None:
            cache["rec"] = S.detach()

        o = self.norm(o.transpose(1, 2).reshape(B * T, self.H, self.dv),
                      z.reshape(B * T, self.H, self.dv))
        return self.out_proj(o.reshape(B, T, -1))
