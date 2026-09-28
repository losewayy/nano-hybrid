import torch

from nanohybrid.config import ModelConfig
from nanohybrid.model import NanoLM
from nanohybrid.deltanet import delta_rule_chunk, delta_rule_recurrent

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. chunk vs recurrent equivalence (with and without initial state) ---
B, H, T, K, V = 2, 6, 128, 32, 32
q = torch.randn(B, H, T, K, device=dev)
k = torch.randn(B, H, T, K, device=dev)
v = torch.randn(B, H, T, V, device=dev)
beta = torch.rand(B, H, T, device=dev)
g = -torch.rand(B, H, T, device=dev) * 2.0
o1, s1 = delta_rule_chunk(q, k, v, beta, g)
o2, s2 = delta_rule_recurrent(q, k, v, beta, g)
assert (o1 - o2).abs().max() / o2.abs().max() < 1e-4
assert (s1 - s2).abs().max() < 0.01
print("chunk vs recurrent ok (rel diff %.2e)" % ((o1 - o2).abs().max() / o2.abs().max()).item())

# with initial state
s0 = torch.randn(B, H, K, V, device=dev)
o1, s1 = delta_rule_chunk(q, k, v, beta, g, state=s0)
o2, s2 = delta_rule_recurrent(q, k, v, beta, g, state=s0)
assert (o1 - o2).abs().max() / o2.abs().max() < 1e-4
print("chunk vs recurrent w/ state ok")

# --- 2. param counts & forward ---
for arch, hid in (("gpt", 1024), ("hybrid", 896)):
    cfg = ModelConfig(arch=arch, mlp_hidden=hid)
    m = NanoLM(cfg).to(dev)
    print(f"{arch}: {m.num_params()/1e6:.2f}M params, layers={cfg.layer_types()}")
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len), device=dev)
    loss = torch.nn.functional.cross_entropy(
        m(idx).view(-1, cfg.vocab_size), idx.view(-1))
    loss.backward()

# --- 3. cached decode == full forward (prefill T>1 then T=1 steps) ---
for arch, hid in (("gpt", 1024), ("hybrid", 896)):
    cfg = ModelConfig(arch=arch, mlp_hidden=hid)
    m = NanoLM(cfg).to(dev).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 64), device=dev)
    with torch.no_grad():
        full = m(idx)
        caches = [{} for _ in m.blocks]
        logits = m(idx[:, :32], caches, 0)            # prefill 32
        for t in range(32, 64):                        # decode one by one
            logits = m(idx[:, t:t + 1], caches, t)
        diff = (full[:, -1] - logits[:, -1]).abs().max().item()
        assert diff < 1e-3, f"{arch} decode diff {diff}"
        print(f"{arch}: prefill->decode consistency ok (diff {diff:.2e})")

# --- 4. T>1 chunk on existing cache (rectangular causal path) ---
m = NanoLM(ModelConfig(arch="gpt")).to(dev).eval()
idx = torch.randint(0, 4096, (1, 64), device=dev)
with torch.no_grad():
    full = m(idx)
    caches = [{} for _ in m.blocks]
    m(idx[:, :30], caches, 0)
    logits = m(idx[:, 30:], caches, 30)               # 34-token chunk on history
    diff = (full[:, -1] - logits[:, -1]).abs().max().item()
    assert diff < 1e-3, f"chunked-cache diff {diff}"
    print(f"gpt: chunked-prefill-on-cache ok (diff {diff:.2e})")

# --- 5. generation runs ---
m.generate(idx[:, :8], max_new=32, temperature=0.8, top_k=50)
print("generate ok")
print("ALL CHECKS PASSED")
