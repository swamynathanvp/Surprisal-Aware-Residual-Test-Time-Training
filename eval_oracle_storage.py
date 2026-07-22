"""Oracle-storage upper-bound eval for SR-TTT v2.

Question: with STORAGE solved (needle force-inserted into the cache at eval,
clearly an oracle diagnostic, not the method), how much retrieval do the
trained read-time projections deliver? Decomposes the v2 negative:
  - exact/token acc still ~prior  -> addressing alone is fatal
  - exact/token acc jumps         -> storage policy is the single blocker

Reuses saved checkpoints (model_B-full.pt / model_B-noOracle.pt) and the same
seeded eval set as the main run. No training. Runtime ~30-45 min on a T4 at
N_TRIALS=30 (override with env SRTTT_ORACLE_TRIALS).
"""
import os, glob, math, random
import numpy as np
import torch
import pandas as pd
from torch.amp import autocast

from sr_ttt_v2 import (CFG, DEVICE, OUTPUT_DIR, TEST_LENGTHS, EVAL_DEPTHS,
                       SRTTTModel, TinyStoriesNeedleGenerator, MockWordTokenizer,
                       build_eval_set, answer_pred_logits, wilson_ci,
                       mcnemar_exact_p)

N_TRIALS = int(os.environ.get("SRTTT_ORACLE_TRIALS",
                              "30" if torch.cuda.is_available() else "2"))


def find_checkpoint(name):
    cands = [os.path.join(OUTPUT_DIR, name)]
    cands += sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


@torch.no_grad()
def greedy_generate_oracle(model, config, sample, force_insert):
    input_ids = sample["input_ids"]
    answer_start, needle_len = sample["answer_start"], sample["needle_len"]
    np_ = sample["needle_positions"].to(DEVICE)
    prefix = input_ids[:, :answer_start].clone()
    generated = []
    for _ in range(needle_len):
        model.reset_caches()
        with autocast('cuda', dtype=config.torch_dtype, enabled=config.use_amp):
            logits, _ = model(prefix.to(DEVICE), start_pos=0, use_cache=True,
                              needle_positions=np_, force_insert=force_insert)
        nxt = logits[0, -1].argmax().view(1, 1).cpu()
        generated.append(nxt.item())
        prefix = torch.cat([prefix, nxt], dim=1)
    return torch.tensor(generated, dtype=torch.long)


@torch.no_grad()
def evaluate_oracle_storage(model, config, eval_set, label, force_insert=True):
    model.eval()
    rows, per_trial = [], {}
    tag = "ORACLE-STORAGE" if force_insert else "filter-only"
    for (sl, d), samples in eval_set.items():
        n_exact, tok_acc, contain_sum, addr_sum = 0, 0.0, 0.0, 0.0
        trial_bits = []
        for s in samples:
            model.reset_caches()
            collect = {}
            with autocast('cuda', dtype=config.torch_dtype, enabled=config.use_amp):
                logits, _ = model(s["input_ids"].to(DEVICE), start_pos=0,
                                  use_cache=True,
                                  needle_positions=s["needle_positions"].to(DEVICE),
                                  answer_positions=s["answer_positions"].to(DEVICE),
                                  force_insert=force_insert, collect=collect)
            preds = answer_pred_logits(logits, s["answer_start"],
                                       s["needle_len"]).argmax(-1).cpu()
            tok_acc += (preds == s["needle_tokens"]).float().mean().item()
            if collect.get("containment"):
                contain_sum += float(np.mean(collect["containment"]))
            if collect.get("oracle_mass"):
                addr_sum += torch.cat([m.float().flatten() for m in
                                       collect["oracle_mass"]]).mean().item()
            gen = greedy_generate_oracle(model, config, s, force_insert)
            em = int(torch.equal(gen, s["needle_tokens"]))
            n_exact += em
            trial_bits.append(em)
            del logits
        n = len(samples)
        lo, hi = wilson_ci(n_exact, n)
        rows.append({"model": label, "mode": tag, "length": sl, "depth": d,
                     "exact": n_exact / n, "exact_lo": lo, "exact_hi": hi,
                     "token_acc": tok_acc / n, "contain": contain_sum / n,
                     "addr_mass": addr_sum / n, "n": n})
        per_trial[(sl, d)] = trial_bits
        print(f"  [{label}|{tag}] len={sl:>5} depth={d:.2f} -> "
              f"exact={n_exact/n:.0%} [{lo:.0%},{hi:.0%}] "
              f"tok_acc={tok_acc/n:.1%} contain={contain_sum/n:.0%} "
              f"addr={addr_sum/n:.3f}")
    return pd.DataFrame(rows), per_trial


def main():
    print(f"Oracle-storage upper-bound eval | trials/cell={N_TRIALS} "
          f"| lengths={TEST_LENGTHS} depths={EVAL_DEPTHS}")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = 65536
    except Exception as e:
        print(f"[WARN] GPT-2 tokenizer unavailable ({type(e).__name__}); "
              f"mock fallback — smoke only")
        tokenizer = MockWordTokenizer(CFG.vocab_size)
    try:
        from datasets import load_dataset
        stories = load_dataset("roneneldan/TinyStories", split="train")["text"]
    except Exception as e:
        print(f"[WARN] TinyStories unavailable; synthetic fallback")
        stories = ["Once upon a time there was a little girl named Lily. "
                   "She loved to play in the garden. " * 20] * 1000

    gen = TinyStoriesNeedleGenerator(CFG, tokenizer, stories)
    eval_set = build_eval_set(gen, TEST_LENGTHS, EVAL_DEPTHS, n_trials=N_TRIALS)

    dfs, trials = [], {}
    for name in ["model_B-full.pt", "model_B-noOracle.pt"]:
        ckpt = find_checkpoint(name)
        if ckpt is None:
            print(f"[SKIP] {name} not found (attach it as a Kaggle dataset)")
            continue
        label = name.replace("model_", "").replace(".pt", "")
        m = SRTTTModel(CFG).to(DEVICE)
        state = torch.load(ckpt, map_location=DEVICE)
        m.load_state_dict(state["model_state_dict"])
        print(f"\n=== {label} with ORACLE STORAGE (force-insert at eval) ===")
        df, tr = evaluate_oracle_storage(m, CFG, eval_set, label, force_insert=True)
        dfs.append(df); trials[label] = tr
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(trials) == 2:
        names = list(trials.keys())
        a, b = trials[names[0]], trials[names[1]]
        B_tot = C_tot = 0
        for key in a:
            B_tot += sum(1 for x, y in zip(a[key], b[key]) if x == 1 and y == 0)
            C_tot += sum(1 for x, y in zip(a[key], b[key]) if x == 0 and y == 1)
        print(f"\nPooled McNemar {names[0]} vs {names[1]} (oracle storage): "
              f"{names[0]}-only={B_tot} {names[1]}-only={C_tot} "
              f"p={mcnemar_exact_p(B_tot, C_tot):.4f}")

    if dfs:
        out = pd.concat(dfs, ignore_index=True)
        out.to_csv(os.path.join(OUTPUT_DIR, "oracle_storage_results.csv"),
                   index=False)
        print(f"\n[OK] Saved -> {OUTPUT_DIR}/oracle_storage_results.csv")


if __name__ == "__main__":
    main()
