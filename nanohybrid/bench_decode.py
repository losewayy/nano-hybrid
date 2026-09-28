"""Decode benchmark: tok/s and resident cache bytes vs context length.

Protocol per (arch, ctx): warmup run, then `reps` timed runs, report median.
  - prefill ctx tokens (not measured)
  - decode `gen` tokens one at a time with caches; first decoded token comes
    from the prefill logits (not a re-fed prompt token)
  - resident_cache_mb = bytes held in kv/conv/rec caches (the tensors that must
    persist across decode steps). This is NOT peak VRAM — transient attention
    workspace and activations are excluded by design.

fp32 weights/caches throughout — absolute numbers are ~2x a bf16 deployment;
the slope comparison is what matters.
"""

import argparse
import json
import os
import statistics
import time

import numpy as np
import torch

from .config import ModelConfig
from .model import NanoLM


def load(arch, ckpt_dir, data_dir, seed):
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    cfg = ModelConfig(arch=arch, mlp_hidden=896 if arch == "hybrid" else 1024)
    cfg.vocab_size = meta["vocab_size"]
    m = NanoLM(cfg).cuda().eval()
    ck = torch.load(os.path.join(ckpt_dir, arch, f"seed{seed}", "last.pt"),
                    map_location="cuda", weights_only=False)
    m.load_state_dict(ck["model"])
    return m


def resident_cache_bytes(caches):
    n = 0
    for c in caches:
        kv = c.get("kv")
        if kv:
            for key in ("k", "v"):
                if kv.get(key) is not None:
                    n += kv[key].numel() * kv[key].element_size()
        for key in ("conv", "rec"):
            if c.get(key) is not None:
                n += c[key].numel() * c[key].element_size()
    return n


@torch.no_grad()
def bench_one(model, ctx, gen, batch):
    idx = torch.randint(0, model.cfg.vocab_size, (batch, ctx), device="cuda")
    caches = [{} for _ in model.blocks]
    logits = model(idx, caches, 0)                       # prefill (not measured)
    cur = logits[:, -1].argmax(-1, keepdim=True)          # first sampled token
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(gen):
        logits = model(cur, caches, ctx + i)
        cur = logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return gen * batch / dt, resident_cache_bytes(caches) / 2**20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=r"F:\projects\_scratch\nano-hybrid\ckpt")
    p.add_argument("--data", default=r"F:\projects\_scratch\nano-hybrid\data")
    p.add_argument("--gen", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    ctxs = [256, 512, 1024, 2048, 4096]
    results = {}
    for arch in ("gpt", "hybrid"):
        model = load(arch, args.ckpt, args.data, args.seed)
        rows = []
        for ctx in ctxs:
            try:
                bench_one(model, ctx, args.gen, args.batch)     # warmup
                runs = [bench_one(model, ctx, args.gen, args.batch)
                        for _ in range(args.reps)]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows.append({"ctx": ctx, "tok_s": None, "resident_cache_mb": None})
                print(f"{arch:7s} ctx={ctx:5d}  OOM")
                continue
            tps = statistics.median(r[0] for r in runs)
            mem = runs[-1][1]
            rows.append({"ctx": ctx, "tok_s": round(tps, 1),
                         "resident_cache_mb": round(mem)})
            print(f"{arch:7s} ctx={ctx:5d}  {tps:8.1f} tok/s  resident {mem:.0f} MB")
        results[arch] = rows

    out = os.path.join(args.ckpt, "bench_decode.json")
    json.dump(results, open(out, "w"), indent=2)
    print("->", out)


if __name__ == "__main__":
    main()
