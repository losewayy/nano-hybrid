"""Download TinyStories, train a small SentencePiece BPE, pretokenize to .bin.

Outputs under DATA_DIR (default ./data):
  tok4096.model / tok4096.vocab   sentencepiece model
  train.bin / val.bin             uint16 token ids (numpy memmap)
  meta.json
"""

import json
import os
import sys

import numpy as np
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO = "roneneldan/TinyStories"
TRAIN_TXT = "TinyStoriesV2-GPT4-train.txt"
VAL_TXT = "TinyStoriesV2-GPT4-valid.txt"
VOCAB = 4096
SPM_TRAIN_CHARS = 64 * 1024 * 1024  # 64MB slice is plenty for BPE training
# cap how much we tokenize (4000 steps * 32 * 256 = ~33M tokens needed; keep margin)
MAX_TRAIN_TOKENS = 45_000_000

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "data"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    train_txt = hf_hub_download(REPO, TRAIN_TXT, repo_type="dataset")
    val_txt = hf_hub_download(REPO, VAL_TXT, repo_type="dataset")
    print("train:", train_txt, "\nval:  ", val_txt)

    # ---- tokenizer ----
    model_prefix = os.path.join(DATA_DIR, f"tok{VOCAB}")
    if not os.path.exists(model_prefix + ".model"):
        tmp = os.path.join(DATA_DIR, "_spm_train.txt")
        with open(train_txt, "rb") as fi, open(tmp, "wb") as fo:
            fo.write(fi.read(SPM_TRAIN_CHARS))
        spm.SentencePieceTrainer.train(
            input=tmp, model_prefix=model_prefix, vocab_size=VOCAB,
            model_type="bpe", character_coverage=1.0, byte_fallback=True,
            split_digits=True, num_threads=os.cpu_count(),
        )
        os.remove(tmp)
    sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
    print("vocab size:", sp.vocab_size())

    # ---- tokenize ----
    for name, path, cap in (("train", train_txt, MAX_TRAIN_TOKENS),
                            ("val", val_txt, None)):
        out_path = os.path.join(DATA_DIR, f"{name}.bin")
        if os.path.exists(out_path):
            print(name, "already done")
            continue
        ids = []
        # stream line-chunks; track buffer size incrementally (quadratic sum() bug fixed)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            buf, buf_len = [], 0
            for line in tqdm(f, desc=f"tokenizing {name}"):
                buf.append(line)
                buf_len += len(line)
                if buf_len > 16 * 1024 * 1024:
                    ids.extend(sp.encode("".join(buf)))
                    buf, buf_len = [], 0
                    if cap and len(ids) >= cap:
                        break
            if buf and (not cap or len(ids) < cap):
                ids.extend(sp.encode("".join(buf)))
        if cap:
            ids = ids[:cap]
        arr = np.asarray(ids, dtype=np.uint16)
        arr.tofile(out_path)
        print(name, len(arr), "tokens ->", out_path)

    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump({"vocab_size": sp.vocab_size(), "tokenizer": f"tok{VOCAB}.model"}, f)


if __name__ == "__main__":
    main()
