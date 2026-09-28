"""Training loop: memmap batches, AdamW, warmup+cosine, bf16 autocast."""

import argparse
import json
import os
import time

import numpy as np
import torch

from .config import ModelConfig, TrainConfig
from .model import NanoLM


def get_batch(data, cfg_ctx, bs, device, gen):
    ix = torch.randint(len(data) - cfg_ctx - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i : i + cfg_ctx].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + cfg_ctx].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, eval_xy, device, bf16=True):
    model.eval()
    losses = []
    for x, y in eval_xy:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            loss = torch.nn.functional.cross_entropy(
                model(x).view(-1, model.cfg.vocab_size), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def lr_at(step, cfg: TrainConfig):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    p = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.min_lr_frac * cfg.lr + 0.5 * (1 + np.cos(np.pi * p)) * (1 - cfg.min_lr_frac) * cfg.lr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["gpt", "hybrid"], required=True)
    p.add_argument("--data", default=r"F:\projects\_scratch\nano-hybrid\data")
    p.add_argument("--out", default=r"F:\projects\_scratch\nano-hybrid\ckpt")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    # hybrid GDN layers carry a few % extra params vs attention; shrink its MLP
    # so both architectures land at ~15.4M total (same parameter budget).
    mcfg = ModelConfig(arch=args.arch, mlp_hidden=896 if args.arch == "hybrid" else 1024)
    tcfg = TrainConfig()
    if args.steps:
        tcfg.steps = args.steps
    if args.seed is not None:
        tcfg.seed = args.seed
    torch.manual_seed(tcfg.seed)
    device = "cuda"

    # dedicated generators: identical batch sequence AND identical eval subset
    # across architectures/seeds => paired comparison
    data_gen = torch.Generator().manual_seed(4242)
    eval_gen = torch.Generator().manual_seed(999)

    train = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    mcfg.vocab_size = meta["vocab_size"]

    model = NanoLM(mcfg).to(device)
    print(f"arch={args.arch} params={model.num_params()/1e6:.2f}M layers={mcfg.layer_types()}")

    decay, no_decay = [], []
    for n, pm in model.named_parameters():
        (decay if pm.dim() >= 2 else no_decay).append(pm)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tcfg.lr, betas=(0.9, 0.95), fused=True)

    run_dir = os.path.join(args.out, args.arch, f"seed{tcfg.seed}")
    os.makedirs(run_dir, exist_ok=True)
    log_f = open(os.path.join(run_dir, "log.jsonl"), "w")

    # fixed eval subset, identical for every run/arch/seed
    eval_xy = []
    for _ in range(tcfg.eval_batches):
        eval_xy.append(get_batch(val, mcfg.ctx_len, tcfg.batch_size, device, eval_gen))

    model.train()
    t0 = time.time()
    for step in range(tcfg.steps):
        lr = lr_at(step, tcfg)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train, mcfg.ctx_len, tcfg.batch_size, device, data_gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tcfg.bf16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, mcfg.vocab_size), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()

        if step % 50 == 0:
            rec = {"step": step, "loss": loss.item(), "lr": lr,
                   "tok_s": int(tcfg.batch_size * mcfg.ctx_len * (step + 1) / (time.time() - t0))}
            print(f"step {step:5d} loss {rec['loss']:.4f} lr {lr:.2e} {rec['tok_s']} tok/s")
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()

        if step % tcfg.eval_every == tcfg.eval_every - 1 or step == tcfg.steps - 1:
            vl = evaluate(model, eval_xy, device, tcfg.bf16)
            rec = {"step": step, "val_loss": vl}
            print(f"  >> step {step} val_loss {vl:.4f}")
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            torch.save({"model": model.state_dict(), "cfg": vars(mcfg), "step": step},
                       os.path.join(run_dir, "last.pt"))

    log_f.close()


if __name__ == "__main__":
    main()
