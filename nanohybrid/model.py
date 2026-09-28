"""NanoLM: shared backbone assembling 'attn' / 'gdn' blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .deltanet import GatedDeltaNet
from .layers import GatedAttention, RMSNorm, SwiGLU, build_rope_cache


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, kind: str):
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = GatedAttention(cfg) if kind == "attn" else GatedDeltaNet(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden)

    def forward(self, x, cos, sin, cache=None, pos=0):
        if self.kind == "attn":
            kv = cache.setdefault("kv", {}) if cache is not None else None
            x = x + self.mixer(self.norm1(x), cos, sin, kv_cache=kv, pos=pos)
        else:
            x = x + self.mixer(self.norm1(x), cache=cache)
        return x + self.mlp(self.norm2(x))


class NanoLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, t) for t in cfg.layer_types()])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight           # tied embeddings
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

    def _rope(self, T, device):
        if self.rope_cos is None or self.rope_cos.shape[2] < T:
            # grow with headroom so decode doesn't rebuild the cache every step
            cap = max(T, 1024) if self.rope_cos is None else max(T, self.rope_cos.shape[2] * 2)
            cos, sin = build_rope_cache(cap, self.cfg.head_dim, self.cfg.rope_theta, device)
            self.rope_cos, self.rope_sin = cos, sin
        return self.rope_cos, self.rope_sin

    def forward(self, idx, caches=None, pos=0):
        B, T = idx.shape
        x = self.embed(idx)
        cos, sin = self._rope(pos + T, idx.device)
        for i, blk in enumerate(self.blocks):
            c = caches[i] if caches is not None else None
            x = blk(x, cos, sin, cache=c, pos=pos)
        return self.lm_head(self.norm_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=1.0, top_k=None):
        """Autoregressive decode. Uses per-layer caches; flushes conv/rec state on T>1 chunks."""
        caches = [{} for _ in self.blocks]
        pos = 0
        for _ in range(max_new):
            logits = self(idx if pos == 0 else idx[:, -1:], caches, pos)[:, -1]
            pos += idx.shape[1] if pos == 0 else 1
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
