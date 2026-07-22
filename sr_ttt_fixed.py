import os; os.environ["PYTHONIOENCODING"] = "utf-8"
# %% [markdown]
# # SR-TTT (FIXED): Surprisal-Aware Residual Test-Time Training
#
# Corrected re-implementation of sr_ttt_kaggle.py. The original run's positive
# result was an artifact. Fixes (search for "FIX n" markers):
#
#   FIX 1  Off-by-one label leak: the loss/eval read logits AT the answer
#          positions (target == current input token), so both models learned
#          to COPY the answer that was already in their input, not retrieve it.
#          -> predictions now come from logits[answer_start-1 : answer_start+L-1].
#   FIX 2  TTT within-window future leak: outputs used W_fast already updated
#          on the whole window. -> outputs now use the state from PREVIOUS
#          update-chunks only (chunked, causal at `ttt_update_chunk` lag).
#   FIX 3  Non-causal cache: all surprising tokens were inserted first, then
#          every query attended with is_causal=False (full future access).
#          -> per-window attend-THEN-insert + strict position mask
#             (cache_positions < query position). EMA threshold now updates
#          sequentially window-by-window (no global-quantile future leak).
#   FIX 4  No LM objective existed (answer-only CE). Stage 1 now trains genuine
#          next-token LM on the sequences, as the paper claims.
#   FIX 5  Train/eval mismatch: eval fed 64-token chunks in separate forwards,
#          re-initializing W_fast every chunk (the "baseline" had no memory at
#          all beyond one window). -> eval is a single full-sequence forward.
#   FIX 6  Exact match is now measured by greedy GENERATION of the answer
#          (no teacher forcing); teacher-forced token accuracy (with the
#          corrected shift) is kept as a secondary metric.
#   FIX 7  Paired evaluation: both models see IDENTICAL seeded eval samples;
#          Wilson 95% CIs and McNemar's exact test are reported.
#   FIX 8  Optional position-free cache addressing (store pre-RoPE keys, query
#          with pre-RoPE queries): retrieval shouldn't depend on the RoPE
#          phase difference between needle position and query position.
#   FIX 9  Honest chance lines in plots (1/|V|, not the stale 25%); data
#          generator guarantees exact length (no silent answer-offset shift).

# %% Cell 1: Environment Setup & Dependencies
import subprocess, sys, time, math, random, string
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

def install_if_missing(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or package])

for _pkg in ["torch", "matplotlib", "seaborn", "tqdm", "pandas", "datasets", "transformers"]:
    install_if_missing(_pkg)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    GPU_NAME = torch.cuda.get_device_name(0)
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
else:
    DEVICE = torch.device("cpu"); GPU_NAME = "CPU"; VRAM_GB = 0

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle") else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% Cell 2: Configuration
@dataclass(frozen=True)
class SRTTTConfig:
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 4
    d_head: int = 64
    d_ff: int = 512

    # TTT inner loop
    ttt_lr: float = 0.01
    window_size: int = 64        # cache flag/insert granularity
    ttt_update_chunk: int = 16   # FIX 2: W_fast update granularity (causal lag)

    # Residual cache
    cache_cap: int = 512
    surprisal_beta: float = 0.9
    surprisal_percentile: float = 95.0
    chunk_size: int = 16

    # Fusion gate
    alpha_init: float = 0.05
    alpha_max: float = 0.5

    # FIX 8: position-free content addressing for the cache
    cache_position_free: bool = True

    vocab_size: int = 50257
    max_seq_len: int = 32768
    rope_theta: float = 10000.0
    dtype: str = "float16"

    @property
    def torch_dtype(self):
        return torch.float16 if self.dtype == "float16" else torch.float32

    @property
    def use_amp(self):
        return self.dtype == "float16" and torch.cuda.is_available()


if VRAM_GB >= 14:
    CFG = SRTTTConfig()
    TEST_LENGTHS = [1024, 2048, 4096]
    TRAIN_SEQ_LEN = 2048
else:
    CFG = SRTTTConfig(cache_cap=256, max_seq_len=2048, dtype="float32")
    TEST_LENGTHS = [256, 512, 1024]
    TRAIN_SEQ_LEN = 128

# %% Cell 3: RoPE & Causal TTT-Linear Layer

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 65536, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, positions=None):
    if positions is not None:
        cos, sin = cos[positions], sin[positions]
    else:
        cos, sin = cos[: x.shape[-2]], sin[: x.shape[-2]]
    if x.dim() == 4 and cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(2); sin = sin.unsqueeze(0).unsqueeze(2)
    elif x.dim() == 3 and cos.dim() == 2:
        cos = cos.unsqueeze(0); sin = sin.unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


class TTTLinearLayer(nn.Module):
    """
    Causal chunked TTT-Linear layer.

    FIX 2: the output for update-chunk j is computed with the fast-weight
    state produced by chunks < j (output-then-update). A token can therefore
    never read tokens at or after its own update-chunk through W_fast.
    """

    def __init__(self, config: SRTTTConfig):
        super().__init__()
        self.config = config
        d, h, dh = config.d_model, config.n_heads, config.d_head
        self.W_q = nn.Linear(d, h * dh, bias=False)
        self.W_k = nn.Linear(d, h * dh, bias=False)
        self.W_v = nn.Linear(d, h * dh, bias=False)
        self.W_o = nn.Linear(h * dh, d, bias=False)
        self.W_fast_init = nn.Parameter(torch.zeros(h, dh, dh))
        nn.init.xavier_uniform_(self.W_fast_init.view(h, dh, dh))
        self.ttt_lr = nn.Parameter(torch.full((h,), config.ttt_lr))
        self.rope = RotaryEmbedding(dh, config.max_seq_len, config.rope_theta)
        self.layer_norm = nn.LayerNorm(dh)

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        B, T, D = x.shape
        cfg = self.config
        H, Dh = cfg.n_heads, cfg.d_head

        q = self.W_q(x).view(B, T, H, Dh)
        k = self.W_k(x).view(B, T, H, Dh)
        v = self.W_v(x).view(B, T, H, Dh)
        k_pre_rope = k                      # FIX 8: kept for position-free cache
        q_pre_rope = q

        cos, sin = self.rope(start_pos + T)
        positions = torch.arange(start_pos, start_pos + T, device=x.device)
        q = apply_rotary_pos_emb(q, cos, sin, positions)
        k = apply_rotary_pos_emb(k, cos, sin, positions)

        k_norm = self.layer_norm(k)
        v_norm = self.layer_norm(v)

        # ---- FP16 STABILITY FIX -----------------------------------------
        # The fast-weight recurrence performs T/uc sequential updates
        # (128 for a 2048-token sequence). Under fp16 autocast, growth in
        # W_fast compounds through the recurrence and the squared residual
        # overflows (fp16 max ~65504) -> intermittent NaN losses.
        # Run the entire recurrence in fp32 with autocast disabled, then
        # cast the output back. This is standard for TTT/fast-weight layers.
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            q32 = q.float()
            k32 = k_norm.float()
            v32 = v_norm.float()

            W_fast = self.W_fast_init.float().unsqueeze(0).expand(B, -1, -1, -1)
            outputs, per_token_losses = [], []
            uc = cfg.ttt_update_chunk
            # Stability clamp: for layer-normed keys, ||k||^2 ~ Dh, so the
            # inner quadratic's curvature is ~2*Dh and gradient descent
            # diverges for lr > ~1/Dh. The lr is learnable; keep it in the
            # stable region.
            lr = self.ttt_lr.view(1, H, 1, 1).abs().clamp(max=1.0 / Dh).float()

            for cs in range(0, T, uc):
                ce = min(cs + uc, T)
                clen = ce - cs
                k_c = k32[:, cs:ce].permute(0, 2, 1, 3)   # [B,H,c,Dh]
                v_c = v32[:, cs:ce].permute(0, 2, 1, 3)
                q_c = q32[:, cs:ce].permute(0, 2, 1, 3)

                # FIX 2: OUTPUT FIRST, with the state from previous chunks.
                o_c = torch.matmul(W_fast, q_c.transpose(-1, -2)).transpose(-1, -2)
                outputs.append(o_c.permute(0, 2, 1, 3))

                # Surprisal loss under the current (pre-update) state — causal.
                W_det = W_fast.detach()
                z = torch.matmul(W_det, k_c.transpose(-1, -2)).transpose(-1, -2)
                token_loss = ((z - v_c) ** 2).mean(dim=-1).mean(dim=1)   # [B,c]
                per_token_losses.append(token_loss.detach())

                # THEN update the state for the next chunk.
                residual = (z - v_c).detach()
                grad_W = (2.0 / clen) * torch.matmul(residual.transpose(-1, -2), k_c.detach())
                W_fast = W_det - lr * grad_W    # lr stays in the graph via next chunk's output

            output = torch.cat(outputs, dim=1).reshape(B, T, H * Dh)
        # -----------------------------------------------------------------
        output = self.W_o(output.to(in_dtype))
        per_token_losses = torch.cat(per_token_losses, dim=1)
        return output, per_token_losses, k, v, k_pre_rope, positions, q_pre_rope


# %% Cell 4: ResidualCache (logic unchanged from original)

class ResidualCache(nn.Module):
    def __init__(self, config: SRTTTConfig):
        super().__init__()
        self.config = config
        cap, H, Dh = config.cache_cap, config.n_heads, config.d_head
        self.register_buffer("cache_k", torch.zeros(cap, H, Dh))
        self.register_buffer("cache_v", torch.zeros(cap, H, Dh))
        self.register_buffer("cache_positions", torch.zeros(cap, dtype=torch.long))
        self.register_buffer("cache_surprisals", torch.zeros(cap))
        self.register_buffer("cache_ages", torch.zeros(cap))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        self.total_inserts = 0
        self.total_evictions = 0

    def reset(self):
        for buf in [self.cache_k, self.cache_v, self.cache_positions,
                    self.cache_surprisals, self.cache_ages]:
            buf.zero_()
        self.count.zero_()
        self.total_inserts = 0
        self.total_evictions = 0

    @torch.no_grad()
    def insert(self, keys, values, positions, surprisals):
        N = keys.shape[0]
        if N == 0:
            return
        cap = self.config.cache_cap
        current = self.count.item()
        if current > 0:
            self.cache_ages[:current] += 1
        self.total_inserts += N
        if current + N <= cap:
            sl = slice(current, current + N)
            self.cache_k[sl] = keys; self.cache_v[sl] = values
            self.cache_positions[sl] = positions
            self.cache_surprisals[sl] = surprisals
            self.cache_ages[sl] = 0
            self.count += N
        else:
            self._evict((current + N) - cap)
            current = self.count.item()
            n_ins = min(N, cap - current)
            sl = slice(current, current + n_ins)
            self.cache_k[sl] = keys[:n_ins]; self.cache_v[sl] = values[:n_ins]
            self.cache_positions[sl] = positions[:n_ins]
            self.cache_surprisals[sl] = surprisals[:n_ins]
            self.cache_ages[sl] = 0
            self.count += n_ins

    @torch.no_grad()
    def _evict(self, n: int):
        current = self.count.item()
        if n >= current:
            self.reset(); return
        self.total_evictions += n
        priorities = self.cache_surprisals[:current] / (1.0 + self.cache_ages[:current])
        _, keep = priorities.topk(current - n, largest=True, sorted=False)
        keep, _ = keep.sort()
        for buf in [self.cache_k, self.cache_v, self.cache_positions,
                    self.cache_surprisals, self.cache_ages]:
            buf[:current - n] = buf[keep]
        self.count.fill_(current - n)

    def get_kv(self):
        # Clones, not views: later insert()/_evict() calls mutate the buffers
        # in place, which would corrupt tensors autograd saved for backward
        # when the cache is read multiple times per forward (window loop).
        c = self.count.item()
        return (self.cache_k[:c].clone(), self.cache_v[:c].clone(),
                self.cache_positions[:c].clone())

    def stats(self):
        return {"fill": self.count.item(), "capacity": self.config.cache_cap,
                "total_inserts": self.total_inserts, "total_evictions": self.total_evictions}


# %% Cell 5: Surprisal Filter (P95 EMA; now applied sequentially per window)

class SurprisalFilter:
    def __init__(self, config: SRTTTConfig):
        self.beta = config.surprisal_beta
        self.percentile = config.surprisal_percentile
        self.chunk_size = config.chunk_size
        self.threshold = None

    def reset(self):
        self.threshold = None

    def compute_threshold(self, losses):
        flat = losses.detach().float().flatten()
        p = torch.quantile(flat, self.percentile / 100.0)
        if self.threshold is None:
            self.threshold = p.item()
        else:
            self.threshold = self.beta * self.threshold + (1 - self.beta) * p.item()
        return self.threshold

    def compute_chunk_surprisal(self, losses):
        B, T = losses.shape
        cs = self.chunk_size
        pad = (cs - T % cs) % cs
        padded = F.pad(losses, (0, pad), value=0.0) if pad else losses
        n_chunks = padded.shape[1] // cs
        means = padded.view(B, n_chunks, cs).mean(dim=-1, keepdim=True)
        return means.expand(-1, -1, cs).reshape(B, -1)[:, :T]

    def filter_tokens(self, per_token_loss):
        threshold = self.compute_threshold(per_token_loss)
        chunk_loss = self.compute_chunk_surprisal(per_token_loss)
        return (per_token_loss > threshold) & (chunk_loss > threshold * 0.8)


# %% Cell 6: Cache Attention (causal) & SR-TTT Block

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CacheAttention(nn.Module):
    """
    Attention over the ResidualCache.

    FIX 3: a strict position mask (cache_positions < query_position) makes the
    readout causal regardless of insertion order.
    FIX 8: with cache_position_free=True, keys are stored pre-RoPE and queries
    are used pre-RoPE, so retrieval is content-addressed and does not depend
    on the (never-trained-for) RoPE phase between needle and query positions.
    """

    def __init__(self, config: SRTTTConfig):
        super().__init__()
        self.config = config
        self.W_o = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)
        self.rope = RotaryEmbedding(config.d_head, config.max_seq_len, config.rope_theta)

    def forward(self, q_pre_rope, cache_k, cache_v, cache_positions, query_start_pos=0):
        B, T = q_pre_rope.shape[:2]
        H, Dh = self.config.n_heads, self.config.d_head
        C = cache_k.shape[0]
        if C == 0:
            return torch.zeros(B, T, self.config.d_model,
                               device=q_pre_rope.device, dtype=q_pre_rope.dtype)

        q_positions = torch.arange(query_start_pos, query_start_pos + T,
                                   device=q_pre_rope.device)
        if self.config.cache_position_free:
            q = q_pre_rope                                          # FIX 8
        else:
            cos, sin = self.rope(self.config.max_seq_len)
            q = apply_rotary_pos_emb(q_pre_rope, cos, sin, q_positions)

        # FIX 3: strict causal mask by original position.
        allowed = cache_positions.view(1, 1, 1, C) < q_positions.view(1, 1, T, 1)

        q = q.permute(0, 2, 1, 3)
        k = cache_k.unsqueeze(0).permute(0, 2, 1, 3).expand(B, -1, -1, -1)
        v = cache_v.unsqueeze(0).permute(0, 2, 1, 3).expand(B, -1, -1, -1)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        attn = torch.nan_to_num(attn, nan=0.0)   # rows with no visible entries -> 0
        attn = attn.permute(0, 2, 1, 3).reshape(B, T, H * Dh)
        return self.W_o(attn)


class SRTTTBlock(nn.Module):
    """
    FIX 3: the cache path now runs window-by-window in order:
        (1) attend this window's queries to the cache AS IT WAS (previous
            windows only), (2) flag + insert this window's surprising tokens,
        (3) EMA threshold updates sequentially (no global-quantile leak).
    A token can therefore never read cache entries from its own or later
    windows; combined with the strict position mask this is causal by
    construction.
    """

    def __init__(self, config: SRTTTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ttt = TTTLinearLayer(config)
        self.cache_attn = CacheAttention(config)
        self.cache = ResidualCache(config)
        self.surprisal_filter = SurprisalFilter(config)
        self.gate_param = nn.Parameter(torch.tensor(config.alpha_init))
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    @property
    def alpha(self):
        return torch.clamp(self.gate_param, min=0.0, max=self.config.alpha_max)

    def forward(self, x, start_pos: int = 0, use_cache: bool = True):
        B, T, D = x.shape
        residual = x
        normed = self.norm1(x)
        ttt_out, per_token_loss, keys, values, keys_pre_rope, positions, q_pre = \
            self.ttt(normed, start_pos)

        if use_cache and B == 1:
            store_k = keys_pre_rope if self.config.cache_position_free else keys
            cache_out = torch.zeros_like(ttt_out)
            W = self.config.window_size
            for ws in range(0, T, W):
                we = min(ws + W, T)
                # (1) attend BEFORE inserting this window
                ck, cv, cpos = self.cache.get_kv()
                cache_out[:, ws:we] = self.cache_attn(
                    q_pre[:, ws:we], ck, cv, cpos, start_pos + ws)
                # (2) flag + insert this window's surprising tokens
                mask = self.surprisal_filter.filter_tokens(per_token_loss[:, ws:we])
                idx = mask[0].nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    self.cache.insert(store_k[0, ws + idx].detach(),
                                      values[0, ws + idx].detach(),
                                      positions[ws + idx].detach(),
                                      per_token_loss[0, ws + idx].detach())
            fused = ttt_out + self.alpha * cache_out
        else:
            fused = ttt_out

        x = residual + fused
        x = x + self.ffn(self.norm2(x))
        return x, per_token_loss

    def reset_cache(self):
        self.cache.reset()
        self.surprisal_filter.reset()


# %% Cell 7: Full Model

class SRTTTModel(nn.Module):
    def __init__(self, config: SRTTTConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([SRTTTBlock(config, i) for i in range(config.n_layers)])
        self.norm_out = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, start_pos: int = 0, use_cache: bool = True):
        x = self.embed(input_ids)
        all_losses = []
        for block in self.blocks:
            x, ptl = block(x, start_pos=start_pos, use_cache=use_cache)
            all_losses.append(ptl)
        x = self.norm_out(x)
        return self.lm_head(x), all_losses

    def reset_caches(self):
        for block in self.blocks:
            block.reset_cache()

    def cache_stats(self):
        return [b.cache.stats() for b in self.blocks]


def causality_self_test(config=None, use_cache=True, verbose=True):
    """Perturb the LAST 8 tokens; logits at all earlier positions must be
    bit-identical. Run at startup so a regression can never ship silently."""
    cfg = config or SRTTTConfig(n_layers=2, d_model=64, n_heads=2, d_head=32,
                                d_ff=128, window_size=32, ttt_update_chunk=8,
                                chunk_size=8, cache_cap=64, vocab_size=100,
                                max_seq_len=512, dtype="float32")
    torch.manual_seed(123)
    m = SRTTTModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 128))
    ids2 = ids.clone(); ids2[0, -8:] = torch.randint(0, cfg.vocab_size, (8,))
    outs = []
    for x in (ids, ids2):
        m.reset_caches()
        with torch.no_grad():
            lg, _ = m(x, use_cache=use_cache)
        outs.append(lg)
    max_leak = (outs[0][:, :-8] - outs[1][:, :-8]).abs().max().item()
    assert max_leak == 0.0, f"CAUSALITY VIOLATION: leak={max_leak:.3e}"
    if verbose:
        print(f"[OK] Causality self-test passed (use_cache={use_cache}): "
              f"max |delta logit| before perturbation = {max_leak:.1e}")


causality_self_test(use_cache=True)
causality_self_test(use_cache=False)


# %% Cell 8: TinyStories Needle-in-a-Haystack Data Pipeline

ALPHANUM = string.ascii_uppercase + string.digits


class TinyStoriesNeedleGenerator:
    """
    [story text...] The secret code is X7B9Q2PA. [more story...]
    Question: What is the secret code? Answer: X7B9Q2PA

    FIX 9: sequences are always built at exactly seq_len (rebuild on shortfall)
    so `answer_start` can never silently shift; the answer span is returned
    explicitly and all indexing downstream uses it.
    """

    def __init__(self, config, tokenizer, stories):
        self.cfg = config
        self.tokenizer = tokenizer
        self.stories = stories
        self.needle_char_len = 8
        self._query_prefix = tokenizer.encode(
            " Question: What is the secret code? Answer:", add_special_tokens=False)

    def _make_needle(self, rng):
        return ''.join(rng.choices(ALPHANUM, k=self.needle_char_len))

    def generate(self, seq_len: int, needle_depth: float = 0.5, rng=None) -> Dict:
        rng = rng or random
        while True:
            needle_str = self._make_needle(rng)
            needle_sentence_ids = self.tokenizer.encode(
                f" The secret code is {needle_str}.", add_special_tokens=False)
            answer_ids = self.tokenizer.encode(f" {needle_str}", add_special_tokens=False)
            query_ids = self._query_prefix + answer_ids
            overhead = len(needle_sentence_ids) + len(query_ids)
            haystack_needed = seq_len - overhead
            assert haystack_needed > 0

            haystack_ids, safety = [], 0
            while len(haystack_ids) < haystack_needed and safety < 400:
                story = self.stories[rng.randint(0, len(self.stories) - 1)]
                haystack_ids.extend(self.tokenizer.encode(story, add_special_tokens=False))
                safety += 1
            if len(haystack_ids) < haystack_needed:
                continue  # FIX 9: rebuild instead of padding
            haystack_ids = haystack_ids[:haystack_needed]

            pos = max(0, min(int(needle_depth * len(haystack_ids)), len(haystack_ids)))
            seq = haystack_ids[:pos] + needle_sentence_ids + haystack_ids[pos:] + query_ids
            assert len(seq) == seq_len
            L = len(answer_ids)
            return {
                "input_ids": torch.tensor(seq, dtype=torch.long).unsqueeze(0),
                "needle_tokens": torch.tensor(answer_ids, dtype=torch.long),
                "answer_start": seq_len - L,   # index of first answer token
                "needle_len": L,
                "needle_depth": needle_depth,
                "seq_len": seq_len,
            }


def answer_pred_logits(logits, answer_start, needle_len):
    """FIX 1: prediction for answer token i lives at logit position
    answer_start - 1 + i (made BEFORE that token is visible)."""
    return logits[0, answer_start - 1: answer_start - 1 + needle_len]


# %% Cell 8.5: Training

def train_srttt(model, config, train_seq_len, generator, n_steps=10000, lr=1e-3,
                log_every=100, accum_steps=4, cache_warmup_step=None, label="SR-TTT"):
    """
    FIX 4: Stage 1 is genuine language modeling (full next-token CE) on the
    needle-bearing sequences — retrieval supervision arises naturally at the
    answer positions under the correct shift. Stage 2 freezes the backbone,
    enables the cache, and adds an explicit answer-CE term so the gate has a
    sharp incentive to exploit the cache.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                                  betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(n_steps // accum_steps, 1), eta_min=lr * 0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
    use_cache = False
    loss_hist, ans_loss_hist, acc_hist = [], [], []
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, n_steps + 1):
        if cache_warmup_step and step == cache_warmup_step:
            use_cache = True
            with torch.no_grad():
                for b in model.blocks:
                    b.gate_param.fill_(config.alpha_init)
                    b.cache.reset()
            for p in model.parameters():
                p.requires_grad = False
            for b in model.blocks:
                b.gate_param.requires_grad = True
                for p in b.cache_attn.parameters():
                    p.requires_grad = True
            cur_lr = scheduler.get_last_lr()[0]
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=cur_lr, weight_decay=0.01,
                                          betas=(0.9, 0.95))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max((n_steps - step + 1) // accum_steps, 1),
                eta_min=lr * 0.1)
            scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
            optimizer.zero_grad(set_to_none=True)
            print(f"  *** STAGE 2 at step {step}: backbone frozen, "
                  f"{sum(p.numel() for p in trainable):,} trainable params ***")

        model.reset_caches()
        depth = random.uniform(0.05, 0.95)
        sample = generator.generate(seq_len=train_seq_len, needle_depth=depth)
        input_ids = sample["input_ids"].to(DEVICE)
        needle_tokens = sample["needle_tokens"].to(DEVICE)
        a_start, L = sample["answer_start"], sample["needle_len"]

        with autocast('cuda', dtype=config.torch_dtype, enabled=config.use_amp):
            logits, _ = model(input_ids, start_pos=0, use_cache=use_cache)
            # FIX 4: genuine next-token LM loss over the whole sequence
            lm_loss = F.cross_entropy(logits[0, :-1].float(), input_ids[0, 1:])
            # FIX 1: correctly-shifted answer loss
            ans_logits = answer_pred_logits(logits, a_start, L).float()
            answer_loss = F.cross_entropy(ans_logits, needle_tokens)
            total = lm_loss + (answer_loss if use_cache else 0.0)

        scaler.scale(total / accum_steps).backward()
        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            acc = (ans_logits.argmax(-1) == needle_tokens).float().mean().item()
        loss_hist.append(lm_loss.item())
        ans_loss_hist.append(answer_loss.item())
        acc_hist.append(acc)

        if step % log_every == 0 or step == 1:
            alpha_avg = np.mean([b.alpha.item() for b in model.blocks])
            print(f"  [{label}] step {step:5d} | lm={np.mean(loss_hist[-log_every:]):.3f} "
                  f"| ans={np.mean(ans_loss_hist[-log_every:]):.3f} "
                  f"| ans_acc={np.mean(acc_hist[-log_every:]):.1%} | alpha={alpha_avg:.4f}")
        del input_ids, logits
        if torch.cuda.is_available() and step % 50 == 0:
            torch.cuda.empty_cache()

    model.eval()
    return loss_hist, ans_loss_hist, acc_hist


# %% Cell 10: Paired Evaluation (FIX 5/6/7)

def build_eval_set(generator, lengths, depths, n_trials, seed=1234):
    """FIX 7: one fixed eval set, identical for every model evaluated."""
    rng = random.Random(seed)
    return {(sl, d): [generator.generate(seq_len=sl, needle_depth=d, rng=rng)
                      for _ in range(n_trials)]
            for sl in lengths for d in depths}


@torch.no_grad()
def greedy_generate_answer(model, config, input_ids, answer_start, needle_len, use_cache):
    """FIX 6: true exact match — generate the answer autoregressively.
    Each step re-runs the full prefix so TTT state and cache are rebuilt
    exactly as in a real single pass (state is not carried across forwards)."""
    prefix = input_ids[:, :answer_start].clone()
    generated = []
    for _ in range(needle_len):
        model.reset_caches()
        with autocast('cuda', dtype=config.torch_dtype, enabled=config.use_amp):
            logits, _ = model(prefix.to(DEVICE), start_pos=0, use_cache=use_cache)
        nxt = logits[0, -1].argmax().view(1, 1).cpu()
        generated.append(nxt.item())
        prefix = torch.cat([prefix, nxt], dim=1)
    return torch.tensor(generated, dtype=torch.long)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar on discordant pairs (b: A right/B wrong,
    c: A wrong/B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


@torch.no_grad()
def evaluate_paired(model, config, eval_set, use_cache, label):
    """FIX 5: single full-sequence forward per trial (same regime as training).
    Returns per-trial exact-match records so models can be compared pairwise."""
    model.eval()
    rows, per_trial = [], {}
    for (sl, d), samples in eval_set.items():
        n_exact, tok_acc = 0, 0.0
        trial_bits = []
        for s in samples:
            model.reset_caches()
            with autocast('cuda', dtype=config.torch_dtype, enabled=config.use_amp):
                logits, _ = model(s["input_ids"].to(DEVICE), start_pos=0,
                                  use_cache=use_cache)
            preds = answer_pred_logits(logits, s["answer_start"],
                                       s["needle_len"]).argmax(-1).cpu()
            tok_acc += (preds == s["needle_tokens"]).float().mean().item()
            gen = greedy_generate_answer(model, config, s["input_ids"],
                                         s["answer_start"], s["needle_len"], use_cache)
            em = int(torch.equal(gen, s["needle_tokens"]))
            n_exact += em
            trial_bits.append(em)
            del logits
        n = len(samples)
        lo, hi = wilson_ci(n_exact, n)
        rows.append({"model": label, "length": sl, "depth": d,
                     "exact": n_exact / n, "exact_lo": lo, "exact_hi": hi,
                     "token_acc": tok_acc / n, "n": n})
        per_trial[(sl, d)] = trial_bits
        print(f"  [{label}] len={sl:>5} depth={d:.2f} -> "
              f"exact={n_exact/n:.0%} [{lo:.0%},{hi:.0%}] token_acc={tok_acc/n:.1%}")
    return pd.DataFrame(rows), per_trial


def paired_significance(trials_a, trials_b):
    print("\nPaired McNemar tests (exact match, per cell and pooled):")
    B_tot = C_tot = 0
    for key in trials_a:
        a, b = trials_a[key], trials_b[key]
        disc_b = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
        disc_c = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
        B_tot += disc_b; C_tot += disc_c
        p = mcnemar_exact_p(disc_b, disc_c)
        print(f"  len={key[0]:>5} depth={key[1]:.2f}: A-only={disc_b:2d} "
              f"B-only={disc_c:2d} p={p:.3f}")
    p_all = mcnemar_exact_p(B_tot, C_tot)
    print(f"  POOLED: A-only={B_tot} B-only={C_tot} p={p_all:.4f}")
    return p_all


# %% Cell 9/11/12: Experiment driver

def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"GPU: {GPU_NAME} | config: {CFG.n_layers}L d={CFG.d_model} "
          f"cache={CFG.cache_cap} pos_free={CFG.cache_position_free}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 65536
    try:
        stories = load_dataset("roneneldan/TinyStories", split="train")["text"]
    except Exception as e:
        print(f"[WARN] TinyStories unavailable ({e}); using synthetic fallback")
        stories = ["Once upon a time there was a little girl named Lily. "
                   "She loved to play in the garden. " * 20] * 1000

    gen = TinyStoriesNeedleGenerator(CFG, tokenizer, stories)
    N_STEPS = 10000 if torch.cuda.is_available() else 200
    STAGE2 = 7001 if torch.cuda.is_available() else 141
    N_TRIALS = 50

    print("\n=== MODEL A: Pure TTT (LM objective, no cache) ===")
    model_a = SRTTTModel(CFG).to(DEVICE)
    train_srttt(model_a, CFG, TRAIN_SEQ_LEN, gen, n_steps=N_STEPS, label="Model A")

    print("\n=== MODEL B: Two-Stage SR-TTT ===")
    model_b = SRTTTModel(CFG).to(DEVICE)
    train_srttt(model_b, CFG, TRAIN_SEQ_LEN, gen, n_steps=N_STEPS,
                cache_warmup_step=STAGE2, label="Model B")

    print("\n=== PAIRED EVALUATION (identical seeded samples) ===")
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    eval_set = build_eval_set(gen, TEST_LENGTHS, depths, n_trials=N_TRIALS)
    df_a, tr_a = evaluate_paired(model_a, CFG, eval_set, use_cache=False,
                                 label="Model A (Pure TTT)")
    df_b, tr_b = evaluate_paired(model_b, CFG, eval_set, use_cache=True,
                                 label="Model B (SR-TTT)")
    paired_significance(tr_a, tr_b)

    combined = pd.concat([df_a, df_b], ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "ab_benchmark_results_fixed.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax, df, title, cmap in [(axes[0], df_a, "Model A exact match", "YlOrRd"),
                                (axes[1], df_b, "Model B exact match", "YlGn")]:
        piv = df.pivot(index="depth", columns="length", values="exact")
        im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_title(title); ax.set_xlabel("context length"); ax.set_ylabel("depth")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.0%}", ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046)
    piv_d = (df_b.pivot(index="depth", columns="length", values="exact")
             - df_a.pivot(index="depth", columns="length", values="exact"))
    im = axes[2].imshow(piv_d.values, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    axes[2].set_xticks(range(len(piv_d.columns))); axes[2].set_xticklabels(piv_d.columns)
    axes[2].set_yticks(range(len(piv_d.index))); axes[2].set_yticklabels(piv_d.index)
    axes[2].set_title("Delta (B - A), generation exact match")
    for i in range(piv_d.shape[0]):
        for j in range(piv_d.shape[1]):
            axes[2].text(j, i, f"{piv_d.values[i, j]:+.0%}", ha="center", va="center")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ab_experiment_results_fixed.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    for name, m in [("model_a_fixed.pt", model_a), ("model_b_fixed.pt", model_b)]:
        torch.save({"model_state_dict": m.state_dict()}, os.path.join(OUTPUT_DIR, name))
    print(f"\n[OK] Done. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
