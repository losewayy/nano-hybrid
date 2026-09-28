"""Plot val-loss curves and decode benchmark comparison."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

CKPT = r"F:\projects\_scratch\nano-hybrid\ckpt"
OUT = os.path.join(CKPT, "figures")
os.makedirs(OUT, exist_ok=True)

# --- val loss curves (aggregated over seed runs) ---
import glob

fig, ax = plt.subplots(figsize=(7, 4.5))
for ci, (arch, label) in enumerate((("gpt", "Full attention (baseline)"),
                                    ("hybrid", "Hybrid 3:1 GDN+Attn"))):
    series = []
    for lf in sorted(glob.glob(os.path.join(CKPT, arch, "seed*", "log.jsonl"))):
        pts = [json.loads(l) for l in open(lf)]
        vals = [(p["step"], p["val_loss"]) for p in pts if "val_loss" in p]
        series.append(dict(vals))
        ax.plot(*zip(*vals), alpha=0.25, lw=1, color=f"C{ci}")
    steps = sorted(set().union(*[s.keys() for s in series]))
    mean = [(s, float(np.mean([d[s] for d in series if s in d]))) for s in steps]
    ax.plot(*zip(*mean), "o-", label=label, lw=2, color=f"C{ci}")
ax.set_xlabel("step"); ax.set_ylabel("val loss")
ax.set_ylim(1.6, 4.4); ax.legend()
ax.set_title("Validation loss (faint = individual seeds)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "loss_curves.png"), dpi=150)

# --- decode benchmark ---
bench = json.load(open(os.path.join(CKPT, "bench_decode.json")))
ctxs = [r["ctx"] for r in bench["gpt"]]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
for arch, label in (("gpt", "Full attention"), ("hybrid", "Hybrid 3:1")):
    rows = bench[arch]
    a1.plot(ctxs, [r["tok_s"] for r in rows], "o-", label=label, lw=2)
    a2.plot(ctxs, [r["resident_cache_mb"] for r in rows], "o-", label=label, lw=2)
a1.set_xlabel("context length"); a1.set_ylabel("decode tok/s"); a1.set_title("Decode speed vs context")
a2.set_xlabel("context length"); a2.set_ylabel("resident cache (MB)")
a2.set_title("Resident decode state vs context")
for a in (a1, a2):
    a.legend(); a.set_xscale("log", base=2)
fig.suptitle("KV cache vs fixed-size recurrent state")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "decode_bench.png"), dpi=150)
print("figures ->", OUT)
