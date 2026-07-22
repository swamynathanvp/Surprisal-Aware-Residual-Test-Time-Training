# SR-TTT Does Not Learn Retrieval — Correction & Mechanistic Post-Mortem

> **⚠️ Correction notice (v2).** Version 1 of this project reported that SR-TTT improved
> Needle-in-a-Haystack exact match by **+23%** (depth 0.50) and **+20%** (depth 0.75) at
> 2048-token contexts. **Those results were evaluation artifacts and are retracted.** The
> training loss and metric read logits *at* the answer positions instead of one position
> earlier, turning the task into copying an answer already visible in the model's input;
> the residual cache and the TTT inner loop additionally leaked future information. Under
> a corrected, causality-verified protocol, the advantage vanishes entirely.
>
> 📄 **Paper (v2, supersedes v1):** [arXiv:2603.06642](https://arxiv.org/abs/2603.06642) —
> *SR-TTT Does Not Learn Retrieval: A Correction and Mechanistic Post-Mortem of
> Surprisal-Aware Residual Test-Time Training.*

## TL;DR

Test-Time Training (TTT) language models replace the KV-cache with fast weights updated
during inference — $O(1)$ memory, but catastrophic failure on exact recall. SR-TTT
hypothesized this could be fixed by routing high-surprisal tokens to a sparse
exact-attention **Residual Cache**. This repository now contains the honest answer:

1. **The v1 result was an artifact.** A model trained on data where retrieval is
   *information-theoretically impossible* reaches 100% accuracy under the v1 metric within
   100 steps (`test_copy_leak.py`). Perturbing only the last 8 input tokens changed logits
   at position 0 with the cache enabled (`test_causality_leak.py`).
2. **Corrected, the effect is not detectably present.** Generation exact match is 0% for
   the baseline *and* all cache variants — zero discordant trials in 750 paired
   comparisons per model pair (pooled McNemar p = 1.0).
3. **The failure decomposes into two independently measured bottlenecks:**
   - **Storage:** surprisal gating is position-biased. Reconstruction-loss "surprise"
     needs burn-in, so early-context needles are almost never stored (0–1% containment at
     depth 0.1 vs. 49–66% mid-sequence) — failing exactly where long-context memory
     matters most. This plausibly extends to related surprise-gated designs (e.g., Titans).
   - **Addressing:** with storage solved by an oracle and *trainable* read-time
     projections added, per-slot attention supervision raises addressing mass 2.5×
     (0.06 → 0.15) yet token accuracy is statistically unchanged; retrieval extracts
     ≈0.06 nats of the 2.30-nat needle. Position-free content addressing cannot resolve
     *ordered* slots whose contents are near-interchangeable.

## Corrected results

Paired evaluation, 50 trials/cell, generation-based exact match, identical seeded samples,
lengths {1024, 2048, 4096} × depths {0.1, 0.25, 0.5, 0.75, 0.9}. Chance token accuracy is
10% (digit needles).

| Condition | Model | Exact match | Token acc. (1024 / 2048 / 4096) | McNemar vs. baseline |
| :--- | :--- | :---: | :---: | :---: |
| Natural storage | A (pure TTT) | 0% | 3.0% / 9.7% / 4.3% | — |
| Natural storage | B-full (oracle supervision + curriculum) | 0% | 6.7% / 10.7% / 8.2% | p = 1.0 (0 discordant) |
| Natural storage | B-noOracle (curriculum only) | 0% | 4.6% / 10.2% / 7.1% | p = 1.0 (0 discordant) |
| **Oracle storage** (upper bound) | B-full | 0% | 12.0% / 17.4% / 11.3% | — |
| **Oracle storage** (upper bound) | B-noOracle | 0% | 12.9% / 16.1% / 12.2% | — |

The v1 tables and figures previously shown here measured a position-tuned copy circuit,
not retrieval, and have been removed.

## Repository contents

| File | Purpose |
| :--- | :--- |
| `sr_ttt_v2.py` | **Main corrected experiment.** Causal TTT + causal cache, LM objective, trainable read-time cache projections, per-slot oracle supervision (ablatable), distance curriculum, digit needles, stage-1 checkpointing, paired eval with Wilson CIs + exact McNemar, storage/addressing diagnostics. Causality self-tests run at import. |
| `sr_ttt_fixed.py` | Minimal corrected version of the original architecture (all leak fixes, no v2 upgrades). |
| `eval_oracle_storage.py` | Oracle-storage upper-bound evaluation (force-inserts the needle at eval; diagnostic only). |
| `test_copy_leak.py` | Smoking gun 1: 100% accuracy on retrieval-impossible data under the v1 metric; chance under corrected indexing. |
| `test_causality_leak.py` | Smoking gun 2: measured future-information leaks in the v1 cache (0.21) and TTT window (0.31). |
| `test_v2_retrieval.py` | Pre-flight: verifies the v2 read-path *can* learn retrieval on a small synthetic task (43% vs. 10% chance) before spending GPU time. |
| `orig_core.py` | The original (flawed) v1 model classes, preserved for reproducibility of the flaw analysis. |
| `main_v2.tex` | Paper source (v2). |

## Reproducing

Kaggle (T4, Internet on) or any single-GPU machine:

```bash
python test_v2_retrieval.py     # ~3 min pre-flight; must print [OK]
python sr_ttt_v2.py             # Stage 1 (7k steps) + two Stage-2 variants (3k each) + paired eval
python eval_oracle_storage.py   # upper-bound diagnostic; reuses saved checkpoints
```

`sr_ttt_v2.py` saves `backbone_stage1.pt` after Stage 1; place it in `/kaggle/input/...`
(or the working directory) on later runs to skip Stage-1 training. Every run begins with
causality self-tests that assert **bit-identical logits** at unperturbed positions with the
cache on and off — if a future change reintroduces a leak, the script refuses to run.

## What survives from v1

The motivation. Pure TTT genuinely scores 0% on exact recall at every length and depth
tested — the corrected experiments confirm the problem more rigorously than v1 did. The
architecture description and two-stage curriculum are also unchanged. What is retracted is
the claimed solution and all evidence offered for it.

## Methodological takeaways (the reusable part)

Retrieval claims for recurrent/TTT-style models should: (a) verify causality mechanically
with perturbation self-tests at startup; (b) measure exact match by *generation*, never
teacher-forced argmax; (c) evaluate paired on identical samples with McNemar tests; and
(d) instrument the mechanism — storage, addressing, readout — rather than reporting only
end-task accuracy, since an end-task null cannot tell you which component failed.

## Citation

```bibtex
@misc{swamynathan2026srttt,
  title  = {SR-TTT Does Not Learn Retrieval: A Correction and Mechanistic Post-Mortem
            of Surprisal-Aware Residual Test-Time Training},
  author = {Swamynathan V P},
  year   = {2026},
  note   = {arXiv:2603.06642v2. Supersedes and corrects v1.}
}
```

Please cite **v2**. The v1 claims should not be cited as evidence that surprisal-gated
caching improves exact recall.
