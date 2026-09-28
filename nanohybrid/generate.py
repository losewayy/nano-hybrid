"""Generate sample stories from a checkpoint."""

import argparse
import json
import os

import sentencepiece as spm
import torch

from .config import ModelConfig
from .model import NanoLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True)
    p.add_argument("--prompt", default="Once upon a time,")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--ckpt", default="ckpt")
    p.add_argument("--data", default="data")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    sp = spm.SentencePieceProcessor(model_file=os.path.join(args.data, meta["tokenizer"]))
    cfg = ModelConfig(arch=args.arch, mlp_hidden=896 if args.arch == "hybrid" else 1024)
    cfg.vocab_size = meta["vocab_size"]
    m = NanoLM(cfg).to(dev).eval()
    ck = torch.load(os.path.join(args.ckpt, args.arch, f"seed{args.seed}", "last.pt"),
                    map_location=dev, weights_only=False)
    m.load_state_dict(ck["model"])
    print(f"loaded {args.arch} step {ck['step']} ({m.num_params()/1e6:.2f}M)")

    idx = torch.tensor([sp.encode(args.prompt)], device=dev)
    out = m.generate(idx, args.tokens, temperature=args.temperature, top_k=50)
    print("---")
    print(sp.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
