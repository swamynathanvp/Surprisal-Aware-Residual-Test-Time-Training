import os; os.environ["PYTHONIOENCODING"] = "utf-8"
# %% [markdown]
# # SR-TTT v2: Can the surprisal cache be made to RETRIEVE?
#
# Builds on sr_ttt_fixed.py (all FIX 1-9 causality/metric corrections retained,
# fp32 TTT recurrence retained). The corrected v1 run showed A ~= B: the cache
# stores the needle but frozen W_q/W_k addressing never learns to retrieve it.
# v2 changes (search for "V2-n" markers):
#
#   V2-1  Hidden-state cache + trainable read-time projections. The cache now
#         stores the token's (detached) pre-TTT hidden state. Keys/values/
#         queries for cache attention are computed AT READ TIME by NEW
#         projections (W_qc/W_kc/W_vc/W_o) that are trainable in Stage 2, so
#         gradients shape the addressing geometry on every step. (Previously
#         stored keys were frozen-W_k snapshots -> no gradient path at all.)
#   V2-2  Oracle attention supervision (ablatable): we know which cache slots
#         hold the needle tokens, so an auxiliary loss -log(attention mass on
#         needle slots) at the answer-prediction positions turns near-
#         impossible credit assignment into supervised learning. Train-time
#         only; ablated by the `oracle` variant flag.
#   V2-3  Mechanism diagnostics, train + eval: containment (are needle tokens
#         resident in the cache at query time?) and addressing mass (attention
#         weight the answer queries place on needle slots). These decompose
#         storage failure vs retrieval failure vs readout failure.
#   V2-4  Distance curriculum in Stage 2: needle->query gap annealed from 128
#         tokens to the sequence maximum (with a random-depth mix), instead of
#         constant max-difficulty placement.
#   V2-5  Digit needles: " The pass key is 3 8 2 9 1 7 4 6." Single-digit
#         GPT-2 tokens with strong embeddings, instead of rare BPE shards of
#         an alphanumeric string. (Old style available via needle_style.)
#   V2-6  Stage-1 backbone is trained ONCE, checkpointed, and every Stage-2
#         variant forks from the same checkpoint (rigorous ablation, and much
#         cheaper on Kaggle quota). If backbone_stage1.pt exists in
#         /kaggle/working or an attached /kaggle/input dataset, it is loaded
#         instead of retrained.
#
# Eval protocol unchanged from the fixed version: paired seeded samples,
# generation-based exact match, Wilson CIs, exact McNemar.

# %% Cell 1: Environment Setup & Dependencies
import subprocess, sys, time, math, random, string, glob
from dataclasses import dataclass, field
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

SMOKE = os.environ.get("SRTTT_SMOKE", "0") == "1"   # tiny CPU end-to-end test

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
    ttt_update_chunk: int = 16   # W_fast update granularity (causal lag)

    # Residual cache
    cache_cap: int = 512
    surprisal_beta: float = 0.9
    surprisal_percentile: float = 95.0
    chunk_size: int = 16

    # Fusion gate
    alpha_init: float = 0.05
    alpha_max: float = 0.5

    # V2-5: needle style
    needle_style: str = "digits"   # "digits" | "alnum"
    needle_len_units: int = 8      # 8 digits or 8 alnum chars

    # V2-2: oracle supervision
    oracle_weight: float = 1.0
    force_insert_needle_in_training: bool = True   # train-time scaffold only;
                                                   # NEVER applied at eval

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
    TEST_LENGTHS = [256, 512]
    TRAIN_SEQ_LEN = 192 if SMOKE else 128

# Experiment budget knobs -----------------------------------------------------
N_STEPS_STAGE1 = 7000 if torch.cuda.is_available() else (60 if SMOKE else 200)
N_STEPS_STAGE2 = 3000 if torch.cuda.is_available() else (40 if SMOKE else 100)
N_TRIALS       = 50   if torch.cuda.is_available() else 2
EVAL_DEPTHS    = [0.1, 0.25, 0.5, 0.75, 0.9] if torch.cuda.is_available() else [0.5]

# V2-6: Stage-2 ablation variants, all forked from the same Stage-1 checkpoint.
STAGE2_VARIANTS = [
    {"name": "B-full",     "oracle": True,  "curriculum": True},
    {"name": "B-noOracle", "oracle": False, "curriculum": True},
    # add e.g. {"name": "B-noCurr", "oracle": True, "curriculum": False},
]

# %% Cell 3: RoPE & Causal TTT-Linear Layer (unchanged from fixed version)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 65536, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._cos_cached = None
        self._sin_cached = None
        self._cached_len = 0

    def forward(self, seq_len: int):
        if seq_len > self._cached_len:
            t = torch.arange(seq_len, device=self.inv_freq.device).float()
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
            self._cached_len = seq_len
        return self._cos_cached, self._sin_cached


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, positions):
    # x: [B, T, H, Dh]
    cos = cos[positions].unsqueeze(0).unsqueeze(2).to(x.dtype)
    sin = sin[positions].unsqueeze(0).unsqueeze(2).to(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


class TTTLinearLayer(nn.Module):
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

        cos, sin = self.rope(start_pos + T)
        positions = torch.arange(start_pos, start_pos + T, device=x.device)
        q = apply_rotary_pos_emb(q, cos, sin, positions)
        k = apply_rotary_pos_emb(k, cos, sin, positions)

        k_norm = self.layer_norm(k)
        v_norm = self.layer_norm(v)

        # FP16 STABILITY: the recurrence performs T/uc sequential updates; run
        # it entirely in fp32 with autocast disabled (see fixed-version notes).
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            q32 = q.float()
            k32 = k_norm.float()
            v32 = v_norm.float()

            W_fast = self.W_fast_init.float().unsqueeze(0).expand(B, -1, -1, -1)
            outputs, per_token_losses = [], []
            uc = cfg.ttt_update_chunk
            # lr is learnable; clamp inside the GD stability region (~1/Dh for
            # layer-normed keys) so the outer optimizer cannot push it into
            # divergence.
            lr = self.ttt_lr.view(1, H, 1, 1).abs().clamp(max=1.0 / Dh).float()

            for cs in range(0, T, uc):
                ce = min(cs + uc, T)
                clen = ce - cs
                k_c = k32[:, cs:ce].permute(0, 2, 1, 3)   # [B,H,c,Dh]
                v_c = v32[:, cs:ce].permute(0, 2, 1, 3)
                q_c = q32[:, cs:ce].permute(0, 2, 1, 3)

                # OUTPUT FIRST, with the state from previous chunks (causal).
                o_c = torch.matmul(W_fast, q_c.transpose(-1, -2)).transpose(-1, -2)
                outputs.append(o_c.permute(0, 2, 1, 3))

                # Surprisal under the current (pre-update) state — causal.
                W_det = W_fast.detach()
                z = torch.matmul(W_det, k_c.transpose(-1, -2)).transpose(-1, -2)
                token_loss = ((z - v_c) ** 2).mean(dim=-1).mean(dim=1)   # [B,c]
                per_token_losses.append(token_loss.detach())

                # THEN update the state for the next chunk.
                residual = (z - v_c).detach()
                grad_W = (2.0 / clen) * torch.matmul(residual.transpose(-1, -2), k_c.detach())
                W_fast = W_det - lr * grad_W    # lr stays in graph via next chunk

            output = torch.cat(outputs, dim=1).reshape(B, T, H * Dh)
        output = self.W_o(output.to(in_dtype))
        per_token_losses = torch.cat(per_token_losses, dim=1)
        return output, per_token_losses


# %% Cell 4: V2-1 Hidden-State ResidualCache

class ResidualCache(nn.Module):
    """Stores DETACHED pre-TTT hidden states of selected tokens, their
    original positions, surprisals, ages, and a needle flag (V2-2/V2-3).
    Keys/values are NOT stored; they are computed at read time by trainable
    projections in CacheReadAttention (V2-1)."""

    def __init__(self, config: SRTTTConfig):
        super().__init__()
        self.config = config
        cap, d = config.cache_cap, config.d_model
        self.register_buffer("cache_h", torch.zeros(cap, d))
        self.register_buffer("cache_positions", torch.zeros(cap, dtype=torch.long))
        self.register_buffer("cache_surprisals", torch.zeros(cap))
        self.register_buffer("cache_ages", torch.zeros(cap))
        self.register_buffer("cache_is_needle", torch.zeros(cap, dtype=torch.bool))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        self.total_inserts = 0
        self.total_evictions = 0

    def reset(self):
        for buf in [self.cache_h, self.cache_positions, self.cache_surprisals,
                    self.cache_ages]:
            buf.zero_()
        self.cache_is_needle.fill_(False)
        self.count.zero_()
        self.total_inserts = 0
        self.total_evictions = 0

    @torch.no_grad()
    def insert(self, hidden, positions, surprisals, is_needle):
        N = hidden.shape[0]
        if N == 0:
            return
        cap = self.config.cache_cap
        current = self.count.item()
        if current > 0:
            self.cache_ages[:current] += 1
        self.total_inserts += N
        if current + N > cap:
            self._evict((current + N) - cap)
            current = self.count.item()
            N = min(N, cap - current)
            hidden, positions = hidden[:N], positions[:N]
            surprisals, is_needle = surprisals[:N], is_needle[:N]
        sl = slice(current, current + N)
        self.cache_h[sl] = hidden.float()
        self.cache_positions[sl] = positions
        self.cache_surprisals[sl] = surprisals.float()
        self.cache_ages[sl] = 0
        self.cache_is_needle[sl] = is_needle
        self.count += N

    @torch.no_grad()
    def _evict(self, n: int):
        current = self.count.item()
        if n >= current:
            self.reset(); return
        self.total_evictions += n
        priorities = self.cache_surprisals[:current] / (1.0 + self.cache_ages[:current])
        _, keep = priorities.topk(current - n, largest=True, sorted=False)
        keep, _ = keep.sort()
        for buf in [self.cache_h, self.cache_positions, self.cache_surprisals,
                    self.cache_ages, self.cache_is_needle]:
            buf[:current - n] = buf[keep]
        self.count.fill_(current - n)

    def get_entries(self):
        # Clones, not views: later insert()/_evict() mutate buffers in place,
        # which would corrupt tensors autograd saved for backward.
        c = self.count.item()
        return (self.cache_h[:c].clone(), self.cache_positions[:c].clone(),
                self.cache_is_needle[:c].clone())

    def stats(self):
        c = self.count.item()
        return {"fill": c, "capacity": self.config.cache_cap,
                "needle_slots": int(self.cache_is_needle[:c].sum().item()),
                "total_inserts": self.total_inserts,
                "total_evictions": self.total_evictions}


# %% Cell 5: Surprisal Filter (P95 EMA, sequential per window; unchanged)

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


# %% Cell 6: V2-1 Cache Read Attention & SR-TTT Block

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CacheReadAttention(nn.Module):
    """V2-1: content-addressed attention over cached hidden states with NEW
    trainable projections applied at read time. Position-free by construction
    (no RoPE anywhere in this path). Manual attention (not SDPA) so weights
    are available for oracle supervision and diagnostics (V2-2/V2-3).
    Strict position mask (cache_pos < query_pos) keeps it causal."""

    def __init__(self, config: SRTTTConfig):
        super().__init__()
        self.config = config
        d, h, dh = config.d_model, config.n_heads, config.d_head
        self.W_qc = nn.Linear(d, h * dh, bias=False)
        self.W_kc = nn.Linear(d, h * dh, bias=False)
        self.W_vc = nn.Linear(d, h * dh, bias=False)
        self.W_o = nn.Linear(h * dh, d, bias=False)

    def forward(self, x_query, cache_h, cache_positions, query_start_pos):
        """x_query [B,T,D]; cache_h [C,D]. Returns (out [B,T,D],
        weights [B,H,T,C] or None, allowed [1,1,T,C] or None)."""
        B, T, D = x_query.shape
        H, Dh = self.config.n_heads, self.config.d_head
        C = cache_h.shape[0]
        if C == 0:
            return (torch.zeros(B, T, D, device=x_query.device, dtype=x_query.dtype),
                    None, None)

        q = self.W_qc(x_query).view(B, T, H, Dh).permute(0, 2, 1, 3)        # [B,H,T,Dh]
        ch = cache_h.to(x_query.dtype)
        k = self.W_kc(ch).view(C, H, Dh).permute(1, 0, 2).unsqueeze(0)      # [1,H,C,Dh]
        v = self.W_vc(ch).view(C, H, Dh).permute(1, 0, 2).unsqueeze(0)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)       # [B,H,T,C]
        q_positions = torch.arange(query_start_pos, query_start_pos + T,
                                   device=x_query.device)
        allowed = cache_positions.view(1, 1, 1, C) < q_positions.view(1, 1, T, 1)
        scores = scores.masked_fill(~allowed, float("-inf"))
        w = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        w = torch.nan_to_num(w, nan=0.0)   # rows with no visible entries -> 0

        out = torch.matmul(w, v).permute(0, 2, 1, 3).reshape(B, T, H * Dh)
        return self.W_o(out), w, allowed


class SRTTTBlock(nn.Module):
    """Cache path runs window-by-window: attend to the cache AS IT WAS, THEN
    flag+insert this window's surprising tokens; EMA threshold updates
    sequentially. Combined with the strict position mask this is causal by
    construction (verified by the startup self-test)."""

    def __init__(self, config: SRTTTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ttt = TTTLinearLayer(config)
        self.cache_attn = CacheReadAttention(config)
        self.cache = ResidualCache(config)
        self.surprisal_filter = SurprisalFilter(config)
        self.gate_param = nn.Parameter(torch.tensor(config.alpha_init))
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    @property
    def alpha(self):
        return torch.clamp(self.gate_param, min=0.0, max=self.config.alpha_max)

    def forward(self, x, start_pos: int = 0, use_cache: bool = True,
                needle_positions=None, answer_positions=None,
                force_insert=False, collect: Optional[dict] = None):
        B, T, D = x.shape
        residual = x
        normed = self.norm1(x)
        ttt_out, per_token_loss = self.ttt(normed, start_pos)

        if use_cache and B == 1:
            positions = torch.arange(start_pos, start_pos + T, device=x.device)
            W = self.config.window_size
            chunks = []
            for ws in range(0, T, W):
                we = min(ws + W, T)
                # (1) attend BEFORE inserting this window
                ch, cpos, cneed = self.cache.get_entries()
                out_w, wts, allowed = self.cache_attn(
                    normed[:, ws:we], ch, cpos, start_pos + ws)
                chunks.append(out_w)

                # V2-2/V2-3: per-position oracle mass + diagnostics.
                # Answer query i's target is THE slot holding needle token i
                # (positions align: answer_positions[i] <-> needle_positions[i]).
                if (collect is not None and answer_positions is not None
                        and needle_positions is not None and wts is not None):
                    local = answer_positions - (start_pos + ws)
                    sel = (local >= 0) & (local < (we - ws))
                    if sel.any():
                        npos_sel = needle_positions[sel.nonzero(as_tuple=True)[0]]
                        eq = (cpos.view(1, -1) == npos_sel.view(-1, 1))  # [nsel,C]
                        collect.setdefault("containment", []).append(
                            eq.any(-1).float().mean().item())
                        if eq.any():
                            mass = (wts[:, :, local[sel]]
                                    * eq.view(1, 1, *eq.shape).to(wts.dtype)
                                    ).sum(-1)                    # [B,H,nsel]
                            collect.setdefault("oracle_mass", []).append(mass)

                # (2) flag + insert this window's surprising tokens
                mask = self.surprisal_filter.filter_tokens(per_token_loss[:, ws:we])
                idx = mask[0].nonzero(as_tuple=True)[0]
                if needle_positions is not None:
                    loc = needle_positions - (start_pos + ws)
                    floc = loc[(loc >= 0) & (loc < (we - ws))]
                    if len(floc) > 0 and collect is not None:
                        # V2-3: would the filter have stored the needle on its
                        # own? (force_insert-independent train diagnostic)
                        n_nat = int(torch.isin(floc, idx).sum().item())
                        collect.setdefault("natural_flagged", []).append(
                            (n_nat, len(floc)))
                    if force_insert and len(floc) > 0:
                        idx = torch.unique(torch.cat([idx, floc]))
                if len(idx) > 0:
                    pos_ins = positions[ws + idx]
                    is_needle = (torch.isin(pos_ins, needle_positions)
                                 if needle_positions is not None
                                 else torch.zeros_like(pos_ins, dtype=torch.bool))
                    self.cache.insert(normed[0, ws + idx].detach(),
                                      pos_ins.detach(),
                                      per_token_loss[0, ws + idx].detach(),
                                      is_needle)
            cache_out = torch.cat(chunks, dim=1)
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

    def forward(self, input_ids, start_pos: int = 0, use_cache: bool = True,
                needle_positions=None, answer_positions=None,
                force_insert=False, collect: Optional[dict] = None):
        x = self.embed(input_ids)
        all_losses = []
        for block in self.blocks:
            x, ptl = block(x, start_pos=start_pos, use_cache=use_cache,
                           needle_positions=needle_positions,
                           answer_positions=answer_positions,
                           force_insert=force_insert, collect=collect)
            all_losses.append(ptl)
        x = self.norm_out(x)
        return self.lm_head(x), all_losses

    def reset_caches(self):
        for block in self.blocks:
            block.reset_cache()

    def cache_stats(self):
        return [b.cache.stats() for b in self.blocks]

    def stage2_parameters(self):
        """V2-1: Stage-2 trainables = fusion gates + read-time cache
        projections (the new addressing geometry)."""
        params = []
        for b in self.blocks:
            params.append(b.gate_param)
            params.extend(b.cache_attn.parameters())
        return params


def causality_self_test(config=None, use_cache=True, verbose=True):
    """Perturb the LAST 8 tokens; logits at all earlier positions must be
    bit-identical. Runs at startup so a regression can never ship silently."""
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


# %% Cell 8: TinyStories Needle Data Pipeline (V2-5 digit needles)

ALPHANUM = string.ascii_uppercase + string.digits


class TinyStoriesNeedleGenerator:
    """
    digits (default): [story...] The pass key is 3 8 2 9 1 7 4 6. [story...]
                      Question: What is the pass key? Answer: 3 8 2 9 1 7 4 6
    alnum:            same, with an 8-char alphanumeric code (v1 style).

    The needle sentence is built as prefix_ids + answer_ids + suffix_ids
    (never re-tokenized as a whole), so the answer token ids appear VERBATIM
    in the context and their global positions are known exactly (needed for
    oracle supervision and containment diagnostics). Sequences are always
    exactly seq_len; `answer_start` never silently shifts.
    """

    def __init__(self, config, tokenizer, stories):
        self.cfg = config
        self.tokenizer = tokenizer
        self.stories = stories
        if config.needle_style == "digits":
            self._prefix_ids = tokenizer.encode(" The pass key is",
                                                add_special_tokens=False)
            self._query_prefix = tokenizer.encode(
                " Question: What is the pass key? Answer:", add_special_tokens=False)
        else:
            self._prefix_ids = tokenizer.encode(" The secret code is",
                                                add_special_tokens=False)
            self._query_prefix = tokenizer.encode(
                " Question: What is the secret code? Answer:", add_special_tokens=False)
        self._suffix_ids = tokenizer.encode(".", add_special_tokens=False)

    def _make_answer_ids(self, rng):
        if self.cfg.needle_style == "digits":
            digits = [rng.choice("0123456789") for _ in range(self.cfg.needle_len_units)]
            text = " " + " ".join(digits)          # " 3 8 2 9 1 7 4 6"
        else:
            code = ''.join(rng.choices(ALPHANUM, k=self.cfg.needle_len_units))
            text = f" {code}"
        return self.tokenizer.encode(text, add_special_tokens=False)

    def generate(self, seq_len: int, needle_depth: float = 0.5,
                 gap_tokens: Optional[int] = None, rng=None) -> Dict:
        """gap_tokens (V2-4): if given, place the needle so that the distance
        from the END of the needle value to answer_start is ~gap_tokens
        (clamped to what the haystack allows); otherwise use needle_depth."""
        rng = rng or random
        while True:
            answer_ids = self._make_answer_ids(rng)
            L = len(answer_ids)
            needle_sentence_ids = self._prefix_ids + answer_ids + self._suffix_ids
            query_ids = self._query_prefix + answer_ids
            overhead = len(needle_sentence_ids) + len(query_ids)
            haystack_needed = seq_len - overhead
            assert haystack_needed > 0, "seq_len too small for needle+query"

            haystack_ids, safety = [], 0
            while len(haystack_ids) < haystack_needed and safety < 400:
                story = self.stories[rng.randint(0, len(self.stories) - 1)]
                haystack_ids.extend(self.tokenizer.encode(story, add_special_tokens=False))
                safety += 1
            if len(haystack_ids) < haystack_needed:
                continue  # rebuild instead of padding
            haystack_ids = haystack_ids[:haystack_needed]
            hay_len = len(haystack_ids)

            S = len(self._suffix_ids)
            Q = len(self._query_prefix)
            if gap_tokens is not None:
                # gap = S + (hay_len - pos) + Q  ->  pos = hay_len + S + Q - gap
                pos = hay_len + S + Q - int(gap_tokens)
                pos = max(0, min(pos, hay_len))
            else:
                pos = max(0, min(int(needle_depth * hay_len), hay_len))

            seq = (haystack_ids[:pos] + needle_sentence_ids
                   + haystack_ids[pos:] + query_ids)
            assert len(seq) == seq_len
            answer_start = seq_len - L
            needle_value_start = pos + len(self._prefix_ids)
            return {
                "input_ids": torch.tensor(seq, dtype=torch.long).unsqueeze(0),
                "needle_tokens": torch.tensor(answer_ids, dtype=torch.long),
                "answer_start": answer_start,
                "needle_len": L,
                # V2-2/V2-3: global positions of the needle VALUE tokens in
                # context, and of the answer-prediction query positions.
                "needle_positions": torch.arange(needle_value_start,
                                                 needle_value_start + L),
                "answer_positions": torch.arange(answer_start - 1,
                                                 answer_start - 1 + L),
                "gap": hay_len - pos + S + Q,
                "needle_depth": needle_depth,
                "seq_len": seq_len,
            }


def answer_pred_logits(logits, answer_start, needle_len):
    """Prediction for answer token i lives at logit position
    answer_start - 1 + i (made BEFORE that token is visible)."""
    return logits[0, answer_start - 1: answer_start - 1 + needle_len]


# %% Cell 9: Training — Stage 1 (backbone) & Stage 2 (variants)

def train_stage1(model, config, train_seq_len, generator, n_steps,
                 lr=1e-3, log_every=100, accum_steps=4):
    """Genuine next-token LM on needle-bearing sequences, cache OFF.
    This is the shared backbone every Stage-2 variant forks from (V2-6)."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                                  betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(n_steps // accum_steps, 1), eta_min=lr * 0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
    loss_hist, ans_loss_hist, acc_hist = [], [], []
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, n_steps + 1):
        model.reset_caches()
        sample = generator.generate(seq_len=train_seq_len,
                                    needle_depth=random.uniform(0.05, 0.95))
        input_ids = sample["input_ids"].to(DEVICE)
        needle_tokens = sample["needle_tokens"].to(DEVICE)
        a_start, L = sample["answer_start"], sample["needle_len"]

        with autocast('cuda', dtype=config.torch_dtype, enabled=config.use_amp):
            logits, _ = model(input_ids, start_pos=0, use_cache=False)
            lm_loss = F.cross_entropy(logits[0, :-1].float(), input_ids[0, 1:])
            ans_logits = answer_pred_logits(logits, a_start, L).float()
            answer_loss = F.cross_entropy(ans_logits, needle_tokens)  # monitor only

        scaler.scale(lm_loss / accum_steps).backward()
        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            acc = (ans_logits.argmax(-1) == needle_tokens).float().mean().item()
        loss_hist.append(lm_loss.item()); ans_loss_hist.append(answer_loss.item())
        acc_hist.append(acc)
        if step % log_every == 0 or step == 1:
            print(f"  [Stage1] step {step:5d} | lm={np.mean(loss_hist[-log_every:]):.3f} "
                  f"| ans={np.mean(ans_loss_hist[-log_every:]):.3f} "
                  f"| ans_acc={np.mean(acc_hist[-log_every:]):.1%}")
        del input_ids, logits
        if torch.cuda.is_available() and step % 50 == 0:
            torch.cuda.empty_cache()
    model.eval()


def train_stage2(model, config, train_seq_len, generator, n_steps, variant,
                 lr=1e-3, log_every=100, accum_steps=4):
    """V2: backbone frozen; trainables = fusion gates + read-time cache
    projections. Loss = LM + answer CE (+ oracle attention loss if enabled).
    Distance curriculum (V2-4) anneals the needle->query gap 128 -> max over
    the first 80% of steps, with a 20% random-depth mix throughout."""
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    trainable = model.stage2_parameters()
    for p in trainable:
        p.requires_grad = True
    with torch.no_grad():
        for b in model.blocks:
            b.gate_param.fill_(config.alpha_init)
    model.reset_caches()

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01,
                                  betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(n_steps // accum_steps, 1), eta_min=lr * 0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
    n_tr = sum(p.numel() for p in trainable)
    print(f"  [{variant['name']}] Stage 2: {n_tr:,} trainable params "
          f"(oracle={variant['oracle']}, curriculum={variant['curriculum']})")

    # curriculum bounds
    S = len(generator._suffix_ids); Q = len(generator._query_prefix)
    min_gap = 128
    max_gap = train_seq_len - 80   # ~ S + Q + hay_len upper bound, with margin

    hists = {"lm": [], "ans": [], "acc": [], "oracle": [], "contain": [],
             "addr": [], "nat": []}
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, n_steps + 1):
        model.reset_caches()
        if variant["curriculum"] and random.random() > 0.2:
            frac = min(1.0, step / max(1, int(0.8 * n_steps)))
            gap = int(min_gap + frac * (max_gap - min_gap))
            gap = int(gap * random.uniform(0.8, 1.2))
            sample = generator.generate(seq_len=train_seq_len, gap_tokens=gap)
        else:
            sample = generator.generate(seq_len=train_seq_len,
                                        needle_depth=random.uniform(0.05, 0.95))
        input_ids = sample["input_ids"].to(DEVICE)
        needle_tokens = sample["needle_tokens"].to(DEVICE)
        a_start, L = sample["answer_start"], sample["needle_len"]
        np_ = sample["needle_positions"].to(DEVICE)
        ap_ = sample["answer_positions"].to(DEVICE)

        collect = {}
        with autocast('cuda', dtype=config.torch_dtype, enabled=config.use_amp):
            logits, _ = model(input_ids, start_pos=0, use_cache=True,
                              needle_positions=np_, answer_positions=ap_,
                              force_insert=config.force_insert_needle_in_training,
                              collect=collect)
            lm_loss = F.cross_entropy(logits[0, :-1].float(), input_ids[0, 1:])
            ans_logits = answer_pred_logits(logits, a_start, L).float()
            answer_loss = F.cross_entropy(ans_logits, needle_tokens)
            total = lm_loss + answer_loss
            oracle_val = 0.0
            if variant["oracle"] and collect.get("oracle_mass"):
                mass = torch.cat([m.float().flatten() for m in collect["oracle_mass"]])
                oracle_loss = -(mass + 1e-6).log().mean()
                total = total + config.oracle_weight * oracle_loss
                oracle_val = oracle_loss.item()

        scaler.scale(total / accum_steps).backward()
        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            acc = (ans_logits.argmax(-1) == needle_tokens).float().mean().item()
            contain = (np.mean(collect["containment"])
                       if collect.get("containment") else 0.0)
            addr = (torch.cat([m.float().flatten() for m in
                               collect["oracle_mass"]]).mean().item()
                    if collect.get("oracle_mass") else 0.0)
            nat = collect.get("natural_flagged", [])
            nat_frac = (sum(a for a, _ in nat) / max(1, sum(b for _, b in nat)))
        hists["lm"].append(lm_loss.item()); hists["ans"].append(answer_loss.item())
        hists["acc"].append(acc); hists["oracle"].append(oracle_val)
        hists["contain"].append(contain); hists["addr"].append(addr)
        hists["nat"].append(nat_frac)

        if step % log_every == 0 or step == 1:
            alpha_avg = np.mean([b.alpha.item() for b in model.blocks])
            k = log_every
            print(f"  [{variant['name']}] step {step:5d} "
                  f"| lm={np.mean(hists['lm'][-k:]):.3f} "
                  f"| ans={np.mean(hists['ans'][-k:]):.3f} "
                  f"| ans_acc={np.mean(hists['acc'][-k:]):.1%} "
                  f"| alpha={alpha_avg:.4f} "
                  f"| contain={np.mean(hists['contain'][-k:]):.0%} "
                  f"| nat_flag={np.mean(hists['nat'][-k:]):.0%} "
                  f"| addr_mass={np.mean(hists['addr'][-k:]):.3f} "
                  f"| oracle={np.mean(hists['oracle'][-k:]):.3f}")
        del input_ids, logits
        if torch.cuda.is_available() and step % 50 == 0:
            torch.cuda.empty_cache()
    model.eval()


# %% Cell 10: Paired Evaluation (protocol unchanged; diagnostics added)

def build_eval_set(generator, lengths, depths, n_trials, seed=1234):
    rng = random.Random(seed)
    return {(sl, d): [generator.generate(seq_len=sl, needle_depth=d, rng=rng)
                      for _ in range(n_trials)]
            for sl in lengths for d in depths}


@torch.no_grad()
def greedy_generate_answer(model, config, sample, use_cache):
    """True exact match: generate autoregressively; each step re-runs the full
    prefix so TTT state and cache are rebuilt exactly as in a single pass.
    NOTE: no force_insert at eval, ever — the surprisal filter must find the
    needle on its own (containment diagnostic reports whether it did)."""
    input_ids = sample["input_ids"]
    answer_start, needle_len = sample["answer_start"], sample["needle_len"]
    np_ = sample["needle_positions"].to(DEVICE)
    prefix = input_ids[:, :answer_start].clone()
    generated = []
    for _ in range(needle_len):
        model.reset_caches()
        with autocast('cuda', dtype=config.torch_dtype, enabled=config.use_amp):
            logits, _ = model(prefix.to(DEVICE), start_pos=0, use_cache=use_cache,
                              needle_positions=np_)
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
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


@torch.no_grad()
def evaluate_paired(model, config, eval_set, use_cache, label):
    model.eval()
    rows, per_trial = [], {}
    for (sl, d), samples in eval_set.items():
        n_exact, tok_acc, contain_sum, addr_sum = 0, 0.0, 0.0, 0.0
        trial_bits = []
        for s in samples:
            model.reset_caches()
            collect = {}
            with autocast('cuda', dtype=config.torch_dtype, enabled=config.use_amp):
                logits, _ = model(s["input_ids"].to(DEVICE), start_pos=0,
                                  use_cache=use_cache,
                                  needle_positions=s["needle_positions"].to(DEVICE),
                                  answer_positions=s["answer_positions"].to(DEVICE),
                                  collect=collect)
            preds = answer_pred_logits(logits, s["answer_start"],
                                       s["needle_len"]).argmax(-1).cpu()
            tok_acc += (preds == s["needle_tokens"]).float().mean().item()
            if collect.get("containment"):
                contain_sum += float(np.mean(collect["containment"]))
            if collect.get("oracle_mass"):
                addr_sum += torch.cat([m.float().flatten() for m in
                                       collect["oracle_mass"]]).mean().item()
            gen = greedy_generate_answer(model, config, s, use_cache)
            em = int(torch.equal(gen, s["needle_tokens"]))
            n_exact += em
            trial_bits.append(em)
            del logits
        n = len(samples)
        lo, hi = wilson_ci(n_exact, n)
        rows.append({"model": label, "length": sl, "depth": d,
                     "exact": n_exact / n, "exact_lo": lo, "exact_hi": hi,
                     "token_acc": tok_acc / n, "contain": contain_sum / n,
                     "addr_mass": addr_sum / n, "n": n})
        per_trial[(sl, d)] = trial_bits
        print(f"  [{label}] len={sl:>5} depth={d:.2f} -> "
              f"exact={n_exact/n:.0%} [{lo:.0%},{hi:.0%}] "
              f"tok_acc={tok_acc/n:.1%} contain={contain_sum/n:.0%} "
              f"addr={addr_sum/n:.3f}")
    return pd.DataFrame(rows), per_trial


def paired_significance(trials_a, trials_b, name_a="A", name_b="B"):
    print(f"\nPaired McNemar (exact match): {name_a} vs {name_b}")
    B_tot = C_tot = 0
    for key in trials_a:
        a, b = trials_a[key], trials_b[key]
        disc_b = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
        disc_c = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
        B_tot += disc_b; C_tot += disc_c
        p = mcnemar_exact_p(disc_b, disc_c)
        print(f"  len={key[0]:>5} depth={key[1]:.2f}: {name_a}-only={disc_b:2d} "
              f"{name_b}-only={disc_c:2d} p={p:.3f}")
    p_all = mcnemar_exact_p(B_tot, C_tot)
    print(f"  POOLED: {name_a}-only={B_tot} {name_b}-only={C_tot} p={p_all:.4f}")
    return p_all


# %% Cell 11: Experiment driver

def find_stage1_checkpoint():
    candidates = [os.path.join(OUTPUT_DIR, "backbone_stage1.pt")]
    candidates += sorted(glob.glob("/kaggle/input/**/backbone_stage1.pt",
                                   recursive=True))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


class MockWordTokenizer:
    """Offline fallback (no-internet sessions / smoke tests): deterministic
    word-level ids via crc32. NOT for real experiments."""

    def __init__(self, vocab_size):
        import zlib
        self.vocab_size = vocab_size
        self._crc = zlib.crc32

    def encode(self, text, add_special_tokens=False):
        return [(self._crc(w.encode()) % (self.vocab_size - 1)) + 1
                for w in text.strip().split()]


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"GPU: {GPU_NAME} | config: {CFG.n_layers}L d={CFG.d_model} "
          f"cache={CFG.cache_cap} needle={CFG.needle_style} "
          f"| stage1={N_STEPS_STAGE1} stage2={N_STEPS_STAGE2} "
          f"variants={[v['name'] for v in STAGE2_VARIANTS]}")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = 65536
    except Exception as e:
        print(f"[WARN] GPT-2 tokenizer unavailable ({type(e).__name__}); "
              f"using offline word-level fallback — smoke/debug only")
        tokenizer = MockWordTokenizer(CFG.vocab_size)
    try:
        stories = load_dataset("roneneldan/TinyStories", split="train")["text"]
    except Exception as e:
        print(f"[WARN] TinyStories unavailable ({e}); using synthetic fallback")
        stories = ["Once upon a time there was a little girl named Lily. "
                   "She loved to play in the garden. " * 20] * 1000

    gen = TinyStoriesNeedleGenerator(CFG, tokenizer, stories)

    # ---- Stage 1: shared backbone (train once, checkpoint, reuse) ----------
    ckpt = find_stage1_checkpoint()
    model_a = SRTTTModel(CFG).to(DEVICE)
    if ckpt:
        print(f"\n=== STAGE 1: loading backbone from {ckpt} ===")
        state = torch.load(ckpt, map_location=DEVICE)
        missing, unexpected = model_a.load_state_dict(
            state["model_state_dict"], strict=False)
        if missing:
            print(f"  (new v2 params initialized fresh: {len(missing)} tensors)")
        model_a.eval()
    else:
        print("\n=== STAGE 1: training shared backbone (LM, cache off) ===")
        train_stage1(model_a, CFG, TRAIN_SEQ_LEN, gen, n_steps=N_STEPS_STAGE1)
        torch.save({"model_state_dict": model_a.state_dict()},
                   os.path.join(OUTPUT_DIR, "backbone_stage1.pt"))
        print(f"  [OK] backbone saved -> {OUTPUT_DIR}/backbone_stage1.pt")

    # ---- Stage 2: fork variants from the same backbone ---------------------
    backbone_state = {k: v.clone() for k, v in model_a.state_dict().items()}
    variant_models = {}
    for variant in STAGE2_VARIANTS:
        print(f"\n=== STAGE 2 [{variant['name']}] ===")
        m = SRTTTModel(CFG).to(DEVICE)
        m.load_state_dict(backbone_state)
        train_stage2(m, CFG, TRAIN_SEQ_LEN, gen, n_steps=N_STEPS_STAGE2,
                     variant=variant)
        variant_models[variant["name"]] = m
        torch.save({"model_state_dict": m.state_dict()},
                   os.path.join(OUTPUT_DIR, f"model_{variant['name']}.pt"))

    # ---- Paired evaluation --------------------------------------------------
    print("\n=== PAIRED EVALUATION (identical seeded samples) ===")
    eval_set = build_eval_set(gen, TEST_LENGTHS, EVAL_DEPTHS, n_trials=N_TRIALS)
    dfs, trials = [], {}
    df_a, tr_a = evaluate_paired(model_a, CFG, eval_set, use_cache=False,
                                 label="A-pureTTT")
    dfs.append(df_a); trials["A-pureTTT"] = tr_a
    for name, m in variant_models.items():
        df_v, tr_v = evaluate_paired(m, CFG, eval_set, use_cache=True, label=name)
        dfs.append(df_v); trials[name] = tr_v

    for name in variant_models:
        paired_significance(trials["A-pureTTT"], trials[name], "A", name)
    if len(variant_models) >= 2:
        names = list(variant_models.keys())
        paired_significance(trials[names[0]], trials[names[1]],
                            names[0], names[1])

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "v2_benchmark_results.csv"), index=False)

    # ---- Plots --------------------------------------------------------------
    model_names = ["A-pureTTT"] + list(variant_models.keys())
    n_panels = len(model_names) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    pivots = {}
    for ax, name in zip(axes[:-1], model_names):
        df = combined[combined["model"] == name]
        piv = df.pivot(index="depth", columns="length", values="exact")
        pivots[name] = piv
        im = ax.imshow(piv.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_title(f"{name} exact match (generation)")
        ax.set_xlabel("context length"); ax.set_ylabel("depth")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.0%}", ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046)
    best = list(variant_models.keys())[0]
    piv_d = pivots[best] - pivots["A-pureTTT"]
    ax = axes[-1]
    im = ax.imshow(piv_d.values, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(piv_d.columns))); ax.set_xticklabels(piv_d.columns)
    ax.set_yticks(range(len(piv_d.index))); ax.set_yticklabels(piv_d.index)
    ax.set_title(f"Delta ({best} - A)")
    for i in range(piv_d.shape[0]):
        for j in range(piv_d.shape[1]):
            ax.text(j, i, f"{piv_d.values[i, j]:+.0%}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "v2_experiment_results.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Done. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
