"""Model / training configs for nano-hybrid.

Two ~15.4M-parameter models, same budget:
  - 'gpt':    N x GatedAttention blocks (modern vanilla transformer)
  - 'hybrid': GatedDeltaNet x3 + GatedAttention x1, repeated (Qwen3-Next style 3:1)
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    arch: str = "gpt"          # 'gpt' | 'hybrid'
    vocab_size: int = 4096
    d_model: int = 384
    n_layer: int = 8
    ctx_len: int = 256

    # attention (Gated Attention, Qwen3-Next flavour: sigmoid gate on attn out,
    # per-head q/k RMSNorm, RoPE, GQA)
    n_head: int = 6
    n_kv_head: int = 2
    head_dim: int = 64
    rope_theta: float = 10000.0
    attn_gate: bool = True     # sigmoid gate on attention output (False => plain vanilla)

    # gated deltanet
    gdn_heads: int = 6         # num key heads = num value heads (ratio 1 for simplicity)
    gdn_head_k: int = 64
    gdn_head_v: int = 64
    gdn_conv_kernel: int = 4

    # mlp
    mlp_hidden: int = 1024     # SwiGLU intermediate size

    # hybrid layout: pattern repeated to fill n_layer
    hybrid_pattern: tuple = ("gdn", "gdn", "gdn", "attn")

    def layer_types(self):
        if self.arch == "gpt":
            return ["attn"] * self.n_layer
        assert self.arch == "hybrid"
        pat = self.hybrid_pattern
        return [pat[i % len(pat)] for i in range(self.n_layer)]


@dataclass
class TrainConfig:
    steps: int = 4000
    batch_size: int = 32       # sequences per step (bs=64 thrashes VRAM allocator on 12GB)
    grad_accum: int = 1
    lr: float = 3e-3
    min_lr_frac: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    seed: int = 1337
    bf16: bool = True
