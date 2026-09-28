"""Shared building blocks: RMSNorm, RoPE, GatedAttention, SwiGLU MLP."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Qwen-style RMSNorm: weight init to 0, applied as (1 + w)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


class RMSNormGated(nn.Module):
    """Per-head RMSNorm followed by a SiLU gate (Qwen3-Next GDN output norm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, gate):
        dtype = x.dtype
        x = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dtype) * F.silu(gate.float())).to(dtype)


def build_rope_cache(ctx_len: int, head_dim: int, theta: float, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(ctx_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                       # [T, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)                # [T, head_dim]
    return emb.cos()[None, None], emb.sin()[None, None]    # [1,1,T,head_dim]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class GatedAttention(nn.Module):
    """GQA attention + per-head q/k RMSNorm + RoPE + optional sigmoid output gate."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.gated = cfg.attn_gate

        q_out = cfg.n_head * cfg.head_dim
        kv_out = cfg.n_kv_head * cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, q_out * (2 if self.gated else 1), bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_out, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin, kv_cache=None, pos: int = 0):
        B, T, _ = x.shape
        qg = self.q_proj(x)
        if self.gated:
            q, gate = qg.chunk(2, dim=-1)
        else:
            q, gate = qg, None
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        c = cos[:, :, pos : pos + T]
        s = sin[:, :, pos : pos + T]
        q, k = apply_rope(q, c, s), apply_rope(k, c, s)

        hist = 0
        if kv_cache is not None:
            # preallocated cache (like real inference engines): geometric growth,
            # write in place — avoids the O(ctx) cat-copy per step.
            hist = kv_cache.get("len", 0)
            buf = kv_cache.get("k")
            need = hist + T
            if buf is None or buf.shape[2] < need:
                new_cap = max(need, (buf.shape[2] if buf is not None else 0) * 2)
                kb = x.new_empty(B, self.n_kv, new_cap, self.head_dim)
                vb = x.new_empty(B, self.n_kv, new_cap, self.head_dim)
                if buf is not None:
                    kb[:, :, :hist], vb[:, :, :hist] = kv_cache["k"], kv_cache["v"]
                kv_cache["k"], kv_cache["v"] = kb, vb
            kv_cache["k"][:, :, hist:need] = k
            kv_cache["v"][:, :, hist:need] = v
            kv_cache["len"] = need
            k = kv_cache["k"][:, :, :need]
            v = kv_cache["v"][:, :, :need]

        kv_len = k.shape[2]
        if T == kv_len:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        elif T == 1:
            # newest token attends to all history
            y = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        else:
            # T>1 chunk on top of history: rectangular causal (bottom-right aligned)
            allowed = torch.arange(kv_len, device=x.device)[None, :] <= (
                hist + torch.arange(T, device=x.device)[:, None])
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed[None, None], enable_gqa=True)
        y = y.transpose(1, 2).reshape(B, T, -1)
        if gate is not None:
            y = y * torch.sigmoid(gate)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
