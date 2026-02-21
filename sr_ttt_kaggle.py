import os; os.environ["PYTHONIOENCODING"] = "utf-8"
# %% [markdown]
# # SR-TTT: Surprisal-Aware Residual Test-Time Training
# **A Novel Hybrid LLM Architecture for Exact-Recall in Long Contexts**
#
# This notebook implements a proof-of-concept for SR-TTT, which augments the
# Test-Time Training (TTT) framework with a dynamic, loss-gated sparse memory
# cache to solve TTT's failure on Needle-in-a-Haystack exact-recall tasks.
#
# **Key Innovations:**
# - Dual-track surprisal filtering (per-token + per-chunk)
# - EMA-smoothed adaptive threshold for cache gating
# - Priority-based eviction (surprisal / (1 + age)) instead of FIFO
# - Learned sigmoid gate for TTT <-> Cache output fusion
# - TinyStories dataset for structured haystack (low-entropy background)

# %% Cell 1: Environment Setup & Dependencies
import subprocess
import sys
import os
import time
import math
import random
import string
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

# Install dependencies (Kaggle-compatible)
def install_if_missing(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or package])

install_if_missing("torch")
install_if_missing("einops")
install_if_missing("matplotlib")
install_if_missing("seaborn")
install_if_missing("tqdm")
install_if_missing("pandas")
install_if_missing("datasets")
install_if_missing("transformers")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Kaggle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from einops import rearrange, repeat, einsum
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# --- Device & Multi-GPU Setup ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    GPU_COUNT = torch.cuda.device_count()
    GPU_NAME = torch.cuda.get_device_name(0)
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[OK] GPU: {GPU_NAME} x {GPU_COUNT} | VRAM: {VRAM_GB:.1f} GB per device")
else:
    DEVICE = torch.device("cpu")
    GPU_COUNT = 0
    GPU_NAME = "CPU"
    VRAM_GB = 0
    print("[!] No CUDA GPU detected -- running on CPU (reduced sequence lengths)")

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# Output directory
OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[OK] Output directory: {OUTPUT_DIR}")
print(f"[OK] PyTorch {torch.__version__} | CUDA {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

# %% Cell 2: Configuration
@dataclass(frozen=True)
class SRTTTConfig:
    """All hyperparameters for the SR-TTT architecture."""
    # Model dimensions
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_head: int = 64           # d_model // n_heads
    d_ff: int = 2048           # FFN intermediate dim (SwiGLU)

    # TTT inner-loop
    ttt_lr: float = 0.01       # Inner-loop SGD learning rate
    ttt_steps: int = 1         # Gradient steps per window
    window_size: int = 256     # Sliding window size for TTT

    # Residual Cache
    cache_cap: int = 2048      # Maximum cached tokens
    surprisal_beta: float = 0.9  # EMA smoothing factor
    surprisal_percentile: float = 95.0  # Percentile threshold (only cache top 5% most surprising)
    chunk_size: int = 64       # Coarse surprisal chunk size

    # Fusion gate
    alpha_init: float = 0.05   # Give cache a strong initial signal (direct param, not sigmoid)
    alpha_max: float = 0.5     # Maximum gate value

    # Vocabulary & sequence
    # GPT-2 tokenizer vocab for real-text experiments
    vocab_size: int = 50257
    max_seq_len: int = 32768
    rope_theta: float = 10000.0

    # Training / evaluation
    dtype: str = "float16"     # "float16" or "float32"

    @property
    def torch_dtype(self):
        return torch.float16 if self.dtype == "float16" else torch.float32

    @property
    def use_amp(self):
        return self.dtype == "float16" and torch.cuda.is_available()


# Adaptive config based on available hardware
if VRAM_GB >= 30:
    # Dual T4 (32 GB total) -- full config
    CFG = SRTTTConfig()
    TEST_LENGTHS = [8192, 16384, 32768]
    TRAIN_SEQ_LEN = 512
elif VRAM_GB >= 14:
    # Single P100 (16 GB) or single/dual T4 (14.6 GB each)
    # Use SMALL model matching successful CPU config -- 11.7M params was
    # hopelessly overparameterized for ~6 bits/example (3 tokens from 4 values).
    # Train with seq_len=256 (4 windows of 64) for good needle SNR,
    # then evaluate on long sequences where cache matters for cross-window retrieval.
    CFG = SRTTTConfig(n_layers=4, d_model=256, n_heads=4, d_head=64, d_ff=512,
                      cache_cap=512, window_size=64, max_seq_len=32768, chunk_size=16)
    TEST_LENGTHS = [1024, 2048, 4096]
    TRAIN_SEQ_LEN = 2048
else:
    # CPU or small GPU -- smoke test config
    CFG = SRTTTConfig(n_layers=4, d_model=256, n_heads=4, d_head=64, d_ff=512,
                      cache_cap=256, window_size=64, max_seq_len=2048, chunk_size=16)
    TEST_LENGTHS = [256, 512, 1024]
    TRAIN_SEQ_LEN = 128

print(f"[OK] Config: {CFG.n_layers}L / d={CFG.d_model} / {CFG.n_heads}H / cache={CFG.cache_cap}")
print(f"[OK] Test context lengths: {TEST_LENGTHS}")

# %% Cell 3: Rotary Positional Embeddings & TTT-Linear Layer

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

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
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, positions=None):
    """Apply rotary embeddings to x at given positions.
    
    x can be [B, T, H, Dh] or [1, C, H, Dh].
    cos/sin are [max_seq_len, Dh].
    After position indexing: [T, Dh] -> reshape to [1, T, 1, Dh] for broadcasting.
    """
    if positions is not None:
        cos = cos[positions]  # [T, Dh]
        sin = sin[positions]  # [T, Dh]
    else:
        cos = cos[: x.shape[-2]]
        sin = sin[: x.shape[-2]]

    # x is [..., seq_len, d_head], cos/sin is [seq_len, d_head]
    # Need to match: add leading dims for batch, and a dim for heads
    if x.dim() == 4:
        # x: [B, T, H, Dh] -> cos needs [1, T, 1, Dh]
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, T, 1, Dh]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, T, 1, Dh]
        elif cos.dim() == 3:
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)
    elif x.dim() == 3:
        # x: [B, T, Dh] -> cos needs [1, T, Dh]
        if cos.dim() == 2:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

    return (x * cos) + (rotate_half(x) * sin)


class TTTLinearLayer(nn.Module):
    """
    Test-Time Training Linear Layer.
    
    Maintains per-head fast-weight matrices W that are updated via gradient
    descent on a self-supervised reconstruction loss during the forward pass.
    Returns both the output AND per-token losses for surprisal filtering.
    """

    def __init__(self, config: SRTTTConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        h = config.n_heads
        dh = config.d_head

        # Projections
        self.W_q = nn.Linear(d, h * dh, bias=False)
        self.W_k = nn.Linear(d, h * dh, bias=False)
        self.W_v = nn.Linear(d, h * dh, bias=False)
        self.W_o = nn.Linear(h * dh, d, bias=False)

        # TTT fast-weight initialization template (learned base)
        self.W_fast_init = nn.Parameter(torch.zeros(h, dh, dh))
        nn.init.xavier_uniform_(self.W_fast_init.view(h, dh, dh))

        # Inner-loop learning rate (learnable per-head)
        self.ttt_lr = nn.Parameter(torch.full((h,), config.ttt_lr))

        self.rope = RotaryEmbedding(dh, config.max_seq_len, config.rope_theta)
        self.layer_norm = nn.LayerNorm(dh)

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        """
        Args:
            x: [B, T, D] input tensor
            start_pos: starting position for RoPE
        Returns:
            output: [B, T, D]
            per_token_losses: [B, T] loss for each token
            keys: [B, T, H, Dh] for cache storage (post-RoPE)
            values: [B, T, H, Dh] for cache storage
            positions: [T] position indices
            q_pre_rope: [B, T, H, Dh] pre-RoPE queries for cache attention
        """
        B, T, D = x.shape
        cfg = self.config
        H, Dh = cfg.n_heads, cfg.d_head

        # Project to Q, K, V
        q = self.W_q(x).view(B, T, H, Dh)
        k = self.W_k(x).view(B, T, H, Dh)
        v = self.W_v(x).view(B, T, H, Dh)
        q_pre_rope = q  # Save pre-RoPE queries for cache attention

        # Apply RoPE
        cos, sin = self.rope(start_pos + T)
        positions = torch.arange(start_pos, start_pos + T, device=x.device)
        q = apply_rotary_pos_emb(q, cos, sin, positions)
        k = apply_rotary_pos_emb(k, cos, sin, positions)

        # Normalize keys and values for stable inner-loop
        k_norm = self.layer_norm(k)
        v_norm = self.layer_norm(v)

        # --- TTT Inner Loop (windowed) ---
        W_fast = self.W_fast_init.unsqueeze(0).expand(B, -1, -1, -1).clone()  # [B, H, Dh, Dh]
        outputs = []
        per_token_losses = []

        window = cfg.window_size
        for win_start in range(0, T, window):
            win_end = min(win_start + window, T)
            win_len = win_end - win_start

            k_win = k_norm[:, win_start:win_end]  # [B, win_len, H, Dh]
            v_win = v_norm[:, win_start:win_end]
            q_win = q[:, win_start:win_end]

            # Reconstruct: z = W_fast @ k  -> compare with v
            # k_win is [B, win_len, H, Dh], W_fast is [B, H, Dh, Dh]
            k_for_matmul = k_win.permute(0, 2, 1, 3)  # [B, H, win_len, Dh]
            v_for_loss = v_win.permute(0, 2, 1, 3)     # [B, H, win_len, Dh]

            # Detach W_fast for the inner-loop update to avoid autograd
            # version conflicts across windowed iterations. The output
            # computation below re-attaches the gradient path.
            W_fast_detached = W_fast.detach()

            # z = W_fast @ k^T -> [B, H, Dh, win_len] -> transpose -> [B, H, win_len, Dh]
            z = torch.matmul(W_fast_detached, k_for_matmul.transpose(-1, -2)).transpose(-1, -2)

            # Per-token reconstruction loss: ||z - v||^2 averaged over Dh
            # (used only for surprisal filtering, detached from training graph)
            token_loss = ((z - v_for_loss) ** 2).mean(dim=-1)  # [B, H, win_len]
            token_loss_avg = token_loss.mean(dim=1)  # [B, win_len] - average over heads
            per_token_losses.append(token_loss_avg.detach())

            # Gradient of loss w.r.t. W_fast (analytical for MSE):
            # dL/dW = 2/N * (z - v) @ k^T = 2/N * residual @ k^T
            residual = (z - v_for_loss).detach()  # [B, H, win_len, Dh]
            grad_W = (2.0 / win_len) * torch.matmul(
                residual.transpose(-1, -2), k_for_matmul.detach()
            )  # [B, H, Dh, Dh]

            # SGD update (on detached W_fast to avoid version conflicts)
            lr = self.ttt_lr.view(1, H, 1, 1).abs()  # ensure positive lr
            W_fast = W_fast_detached - lr * grad_W

            # Output: o = W_fast_updated @ q (this IS in the computation graph
            # since W_fast now depends on lr, which is a learnable parameter)
            q_for_matmul = q_win.permute(0, 2, 1, 3)  # [B, H, win_len, Dh]
            o_win = torch.matmul(
                W_fast, q_for_matmul.transpose(-1, -2)
            ).transpose(-1, -2)  # [B, H, win_len, Dh]
            outputs.append(o_win.permute(0, 2, 1, 3))  # [B, win_len, H, Dh]

        output = torch.cat(outputs, dim=1)  # [B, T, H, Dh]
        per_token_losses = torch.cat(per_token_losses, dim=1)  # [B, T]

        # Merge heads and project
        output = output.reshape(B, T, H * Dh)
        output = self.W_o(output)

        return output, per_token_losses, k, v, positions, q_pre_rope


# --- Shape Assertion Test ---
def test_ttt_linear():
    cfg_test = SRTTTConfig(n_layers=2, d_model=128, n_heads=2, d_head=64,
                           window_size=32, max_seq_len=256, vocab_size=100)
    layer = TTTLinearLayer(cfg_test).to(DEVICE)
    x = torch.randn(2, 64, 128, device=DEVICE)
    with torch.no_grad():
        out, losses, keys, values, pos, q_pre = layer(x)
    assert out.shape == (2, 64, 128), f"Output shape mismatch: {out.shape}"
    assert losses.shape == (2, 64), f"Loss shape mismatch: {losses.shape}"
    assert keys.shape == (2, 64, 2, 64), f"Keys shape mismatch: {keys.shape}"
    print("[OK] TTTLinearLayer shape tests passed")

test_ttt_linear()

# %% Cell 4: ResidualCache Module

class ResidualCache(nn.Module):
    """
    Sparse memory cache for surprising tokens.
    
    Uses priority-based eviction: priority = surprisal / (1 + age).
    Pre-allocates fixed-size tensors to avoid dynamic memory allocation.
    """

    def __init__(self, config: SRTTTConfig):
        super().__init__()
        self.config = config
        cap = config.cache_cap
        H = config.n_heads
        Dh = config.d_head

        # Pre-allocated cache storage (not model parameters -- just buffers)
        self.register_buffer("cache_k", torch.zeros(cap, H, Dh))
        self.register_buffer("cache_v", torch.zeros(cap, H, Dh))
        self.register_buffer("cache_positions", torch.zeros(cap, dtype=torch.long))
        self.register_buffer("cache_surprisals", torch.zeros(cap))
        self.register_buffer("cache_ages", torch.zeros(cap))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

        # Metrics tracking
        self.total_inserts = 0
        self.total_evictions = 0

    def reset(self):
        """Clear the cache for a new sequence."""
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.cache_positions.zero_()
        self.cache_surprisals.zero_()
        self.cache_ages.zero_()
        self.count.zero_()
        self.total_inserts = 0
        self.total_evictions = 0

    @torch.no_grad()
    def insert(self, keys: torch.Tensor, values: torch.Tensor,
               positions: torch.Tensor, surprisals: torch.Tensor):
        """
        Insert surprising tokens into the cache.
        
        Args:
            keys:      [N, H, Dh] - keys of surprising tokens
            values:    [N, H, Dh] - values of surprising tokens
            positions: [N] - original position indices
            surprisals:[N] - surprisal scores
        """
        N = keys.shape[0]
        if N == 0:
            return

        cap = self.config.cache_cap
        current = self.count.item()

        # Age existing entries
        if current > 0:
            self.cache_ages[:current] += 1

        self.total_inserts += N

        if current + N <= cap:
            # Enough room -- direct insert
            self.cache_k[current:current + N] = keys
            self.cache_v[current:current + N] = values
            self.cache_positions[current:current + N] = positions
            self.cache_surprisals[current:current + N] = surprisals
            self.cache_ages[current:current + N] = 0
            self.count += N
        else:
            # Need to evict -- priority-based
            n_to_evict = (current + N) - cap
            self._evict(n_to_evict)
            current = self.count.item()
            n_insert = min(N, cap - current)
            self.cache_k[current:current + n_insert] = keys[:n_insert]
            self.cache_v[current:current + n_insert] = values[:n_insert]
            self.cache_positions[current:current + n_insert] = positions[:n_insert]
            self.cache_surprisals[current:current + n_insert] = surprisals[:n_insert]
            self.cache_ages[current:current + n_insert] = 0
            self.count += n_insert

    @torch.no_grad()
    def _evict(self, n: int):
        """Evict n entries with lowest priority = surprisal / (1 + age)."""
        current = self.count.item()
        if n >= current:
            self.reset()
            return

        self.total_evictions += n

        priorities = self.cache_surprisals[:current] / (1.0 + self.cache_ages[:current])
        # Keep entries with HIGHEST priority
        _, keep_indices = priorities.topk(current - n, largest=True, sorted=False)
        keep_indices, _ = keep_indices.sort()

        # Compact the cache
        self.cache_k[:current - n] = self.cache_k[keep_indices]
        self.cache_v[:current - n] = self.cache_v[keep_indices]
        self.cache_positions[:current - n] = self.cache_positions[keep_indices]
        self.cache_surprisals[:current - n] = self.cache_surprisals[keep_indices]
        self.cache_ages[:current - n] = self.cache_ages[keep_indices]
        self.count.fill_(current - n)

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (keys, values, positions) of cached tokens."""
        c = self.count.item()
        return self.cache_k[:c], self.cache_v[:c], self.cache_positions[:c]

    @property
    def fill_ratio(self) -> float:
        return self.count.item() / self.config.cache_cap

    def stats(self) -> Dict:
        return {
            "fill": self.count.item(),
            "capacity": self.config.cache_cap,
            "fill_ratio": self.fill_ratio,
            "total_inserts": self.total_inserts,
            "total_evictions": self.total_evictions,
        }


# --- Cache Test ---
def test_cache():
    cfg_test = SRTTTConfig(cache_cap=8, n_heads=2, d_head=4, d_model=8)
    cache = ResidualCache(cfg_test).to(DEVICE)

    # Insert 5 tokens
    k = torch.randn(5, 2, 4, device=DEVICE)
    v = torch.randn(5, 2, 4, device=DEVICE)
    p = torch.arange(5, device=DEVICE)
    s = torch.tensor([0.1, 0.5, 0.9, 0.3, 0.7], device=DEVICE)
    cache.insert(k, v, p, s)
    assert cache.count.item() == 5, f"Expected 5, got {cache.count.item()}"

    # Insert 5 more (triggers eviction since cap=8)
    k2 = torch.randn(5, 2, 4, device=DEVICE)
    v2 = torch.randn(5, 2, 4, device=DEVICE)
    p2 = torch.arange(5, 10, device=DEVICE)
    s2 = torch.tensor([0.8, 0.2, 0.6, 0.4, 0.95], device=DEVICE)
    cache.insert(k2, v2, p2, s2)
    assert cache.count.item() == 8, f"Expected 8 after eviction, got {cache.count.item()}"
    print(f"[OK] ResidualCache tests passed | Stats: {cache.stats()}")

test_cache()

# %% Cell 5: Surprisal Filter

class SurprisalFilter:
    """
    Dual-track surprisal filtering with EMA-smoothed threshold.
    
    A token is flagged as "surprising" only if:
      1. Its per-token loss exceeds the EMA-smoothed P99 threshold, AND
      2. The chunk it belongs to has above-threshold mean loss.
    """

    def __init__(self, config: SRTTTConfig):
        self.beta = config.surprisal_beta
        self.percentile = config.surprisal_percentile
        self.chunk_size = config.chunk_size
        self.threshold = None  # Will be initialized on first call

    def reset(self):
        self.threshold = None

    def compute_threshold(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute EMA-smoothed percentile threshold.
        Args: losses [B, T] or [T]
        Returns: scalar threshold
        """
        flat = losses.detach().float().flatten()
        p99 = torch.quantile(flat, self.percentile / 100.0)
        if self.threshold is None:
            self.threshold = p99.item()
        else:
            self.threshold = self.beta * self.threshold + (1 - self.beta) * p99.item()
        return self.threshold

    def compute_chunk_surprisal(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute per-chunk mean loss.
        Args: losses [B, T]
        Returns: chunk_loss [B, T] (each token gets its chunk's mean)
        """
        B, T = losses.shape
        cs = self.chunk_size

        # Pad to multiple of chunk_size
        pad_len = (cs - T % cs) % cs
        if pad_len > 0:
            padded = F.pad(losses, (0, pad_len), value=0.0)
        else:
            padded = losses

        # Reshape into chunks and compute mean
        n_chunks = padded.shape[1] // cs
        chunked = padded.view(B, n_chunks, cs)
        chunk_means = chunked.mean(dim=-1, keepdim=True)  # [B, n_chunks, 1]
        chunk_expanded = chunk_means.expand(-1, -1, cs).reshape(B, -1)[:, :T]  # [B, T]

        return chunk_expanded

    def filter_tokens(self, per_token_loss: torch.Tensor) -> torch.Tensor:
        """
        Dual-track filtering.
        Args: per_token_loss [B, T]
        Returns: mask [B, T] boolean -- True for surprising tokens
        """
        threshold = self.compute_threshold(per_token_loss)
        chunk_loss = self.compute_chunk_surprisal(per_token_loss)

        # Track 1: per-token exceeds threshold
        token_mask = per_token_loss > threshold

        # Track 2: chunk mean exceeds threshold (with a softer multiplier)
        chunk_mask = chunk_loss > (threshold * 0.8)

        # Dual-track: both must be true
        combined = token_mask & chunk_mask
        return combined


# --- Surprisal Filter Test ---
def test_surprisal_filter():
    cfg_test = SRTTTConfig(chunk_size=4, surprisal_percentile=95.0)
    sf = SurprisalFilter(cfg_test)

    # Create losses with 64 tokens - mostly low, one extreme outlier at position 60
    normal_losses = torch.rand(1, 63) * 0.3 + 0.05  # range [0.05, 0.35]
    outlier = torch.tensor([[10.0]])  # Extreme outlier
    losses = torch.cat([normal_losses, outlier], dim=1)  # [1, 64]
    mask = sf.filter_tokens(losses)
    assert mask[0, -1].item() == True, f"Outlier (loss=10.0) should be flagged, threshold={sf.threshold:.4f}"
    assert mask[0, 0].item() == False, "Normal token should not be flagged"
    n_flagged = mask.sum().item()
    print(f"[OK] SurprisalFilter test passed | Threshold: {sf.threshold:.4f} | Flagged: {n_flagged}/64")

test_surprisal_filter()

# %% Cell 6: Hybrid SR-TTT Block

class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CacheAttention(nn.Module):
    """
    Multi-Head Attention restricted to the ResidualCache K/V.
    Shares Q projection with TTTLinearLayer to ensure Q/K alignment.
    Uses F.scaled_dot_product_attention (auto-dispatches to FlashAttn when available).
    """

    def __init__(self, config: SRTTTConfig):
        super().__init__()
        self.config = config
        # No W_q here — queries come pre-projected from TTT's W_q
        self.W_o = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)
        self.rope = RotaryEmbedding(config.d_head, config.max_seq_len, config.rope_theta)

    def forward(self, q_pre_rope: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor,
                cache_positions: torch.Tensor, query_start_pos: int = 0):
        """
        Args:
            q_pre_rope: [B, T, H, Dh] - pre-RoPE queries from TTT's W_q
            cache_k: [C, H, Dh] - cached keys (post-RoPE from TTT's W_k)
            cache_v: [C, H, Dh] - cached values
            cache_positions: [C] - original positions of cached tokens
        Returns:
            output: [B, T, D]
        """
        B, T = q_pre_rope.shape[:2]
        H, Dh = self.config.n_heads, self.config.d_head
        C = cache_k.shape[0]

        if C == 0:
            return torch.zeros(B, T, self.config.d_model, device=q_pre_rope.device,
                               dtype=q_pre_rope.dtype)

        # Apply RoPE to queries (keys are already post-RoPE from TTT)
        cos, sin = self.rope(self.config.max_seq_len)
        q_positions = torch.arange(query_start_pos, query_start_pos + T, device=q_pre_rope.device)
        q = apply_rotary_pos_emb(q_pre_rope, cos, sin, q_positions)

        # Cached keys are already post-RoPE — no need to re-apply RoPE

        # Reshape for attention: [B, H, T, Dh] and [B, H, C, Dh]
        q = q.permute(0, 2, 1, 3)  # [B, H, T, Dh]
        k = cache_k.unsqueeze(0).permute(0, 2, 1, 3).expand(B, -1, -1, -1)  # [B, H, C, Dh]
        v = cache_v.unsqueeze(0).permute(0, 2, 1, 3).expand(B, -1, -1, -1)  # [B, H, C, Dh]

        # Scaled dot-product attention (auto-dispatches to FlashAttn v2 if available)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # [B, H, T, Dh]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, H * Dh)

        return self.W_o(attn_out)


class SRTTTBlock(nn.Module):
    """
    Hybrid SR-TTT Block:
    
    Input -> LN -> TTTLinear -> (surprisal filter -> cache insert)
                            -> CacheAttn -> alpha-gated fusion -> Residual
          -> LN -> SwiGLU FFN -> Residual -> Output
    """

    def __init__(self, config: SRTTTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # TTT path
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ttt = TTTLinearLayer(config)

        # Cache attention path
        self.cache_attn = CacheAttention(config)
        self.cache = ResidualCache(config)
        self.surprisal_filter = SurprisalFilter(config)

        # Fusion gate: alpha = clamp(gate_param, 0, alpha_max)
        self.gate_param = nn.Parameter(torch.tensor(config.alpha_init))

        # FFN
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    @property
    def alpha(self):
        return torch.clamp(self.gate_param, min=0.0, max=self.config.alpha_max)

    def forward(self, x: torch.Tensor, start_pos: int = 0,
                use_cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D]
            start_pos: starting position for RoPE
            use_cache: if False, skip cache path (baseline mode)
        Returns:
            output: [B, T, D]
            per_token_loss: [B, T]
        """
        B, T, D = x.shape
        residual = x

        # TTT path
        normed = self.norm1(x)
        ttt_out, per_token_loss, keys, values, positions, q_pre_rope = self.ttt(normed, start_pos)

        if use_cache and B == 1:  # Cache operates on single sequences
            # Surprisal filtering
            mask = self.surprisal_filter.filter_tokens(per_token_loss)  # [B, T]

            # Extract surprising tokens (batch dim 0 since B=1 for cache)
            surprising_idx = mask[0].nonzero(as_tuple=True)[0]
            if len(surprising_idx) > 0:
                s_keys = keys[0, surprising_idx]        # [N, H, Dh]
                s_values = values[0, surprising_idx]     # [N, H, Dh]
                s_positions = positions[surprising_idx]  # [N]
                s_losses = per_token_loss[0, surprising_idx]  # [N]

                # Keep K/V in graph for gradient flow (don't detach)
                self.cache.insert(s_keys, s_values,
                                  s_positions.detach(), s_losses.detach())

            # Cache attention — uses TTT's Q projection (shared Q/K space)
            cache_k, cache_v, cache_pos = self.cache.get_kv()
            cache_out = self.cache_attn(q_pre_rope, cache_k, cache_v, cache_pos, start_pos)

            # Gated fusion
            alpha = self.alpha
            fused = ttt_out + alpha * cache_out
        else:
            fused = ttt_out

        x = residual + fused

        # FFN
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x, per_token_loss

    def reset_cache(self):
        self.cache.reset()
        self.surprisal_filter.reset()


# --- Block Test ---
def test_srttt_block():
    cfg_test = SRTTTConfig(n_layers=1, d_model=128, n_heads=2, d_head=64,
                           d_ff=256, window_size=32, max_seq_len=256,
                           cache_cap=64, chunk_size=8)
    block = SRTTTBlock(cfg_test, layer_idx=0).to(DEVICE)
    x = torch.randn(1, 64, 128, device=DEVICE)
    with torch.no_grad():
        out, losses = block(x, use_cache=True)
    assert out.shape == (1, 64, 128), f"Block output shape: {out.shape}"
    assert losses.shape == (1, 64), f"Block losses shape: {losses.shape}"
    print(f"[OK] SRTTTBlock test passed | Gate alpha = {block.alpha.item():.6f} | "
          f"Cache: {block.cache.stats()}")

test_srttt_block()

# %% Cell 7: Full SR-TTT Model

class SRTTTModel(nn.Module):
    """
    Full Surprisal-Aware Residual TTT Language Model.
    
    Architecture:
        Token Embedding + RoPE -> N x SRTTTBlock -> LayerNorm -> LM Head
    """

    def __init__(self, config: SRTTTConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            SRTTTBlock(config, layer_idx=i) for i in range(config.n_layers)
        ])
        self.norm_out = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (embedding <-> lm_head)
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0,
                use_cache: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            input_ids: [B, T] token indices
            start_pos: starting position for incremental decoding
            use_cache: enable/disable residual cache
        Returns:
            logits: [B, T, V]
            all_losses: list of [B, T] per-layer per-token losses
        """
        x = self.embed(input_ids)

        all_losses = []
        for block in self.blocks:
            x, per_token_loss = block(x, start_pos=start_pos, use_cache=use_cache)
            all_losses.append(per_token_loss)

        x = self.norm_out(x)
        logits = self.lm_head(x)

        return logits, all_losses

    def reset_caches(self):
        """Reset all layer caches for a new sequence."""
        for block in self.blocks:
            block.reset_cache()

    def cache_stats(self) -> List[Dict]:
        return [block.cache.stats() for block in self.blocks]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- Model Instantiation & VRAM Check ---
def create_model(config, label="Model"):
    """Create and initialize a fresh SRTTTModel."""
    m = SRTTTModel(config).to(DEVICE)
    # DataParallel for dual-GPU setups
    if GPU_COUNT > 1:
        m = nn.DataParallel(m)
        print(f"[OK] {label}: DataParallel enabled across {GPU_COUNT} GPUs")
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[OK] {label}: {n_params / 1e6:.1f}M parameters")
    return m, n_params

# Quick VRAM check with a throwaway model
if torch.cuda.is_available():
    _test_model = SRTTTModel(CFG).to(DEVICE)
    torch.cuda.reset_peak_memory_stats()
    x_test = torch.randint(0, CFG.vocab_size, (1, min(256, CFG.window_size)), device=DEVICE)
    with torch.no_grad():
        _ = _test_model(x_test, use_cache=True)
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"[OK] Peak VRAM per model (warmup): {peak_mem:.2f} GB")
    del _test_model, x_test
    torch.cuda.empty_cache()

# %% Cell 8: TinyStories Needle-in-a-Haystack Data Pipeline

# --- Load dataset and tokenizer ---
print("\n[INFO] Loading TinyStories dataset and GPT-2 tokenizer...")
_TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
_TOKENIZER.pad_token = _TOKENIZER.eos_token
_TOKENIZER.model_max_length = 8192  # Override GPT-2's 1024 limit for long-context eval
try:
    _STORIES_DS = load_dataset("roneneldan/TinyStories", split="train")
    _STORIES_LIST = _STORIES_DS["text"]
    print(f"[OK] TinyStories loaded: {len(_STORIES_LIST)} stories")
except Exception as e:
    print(f"[WARN] Could not load TinyStories: {e}")
    print("[WARN] Falling back to synthetic stories for testing")
    _STORIES_LIST = [
        "Once upon a time there was a little girl named Lily. She loved to play in the garden. " * 20
    ] * 1000

ALPHANUM = string.ascii_uppercase + string.digits  # A-Z, 0-9


class TinyStoriesNeedleGenerator:
    """
    Generates needle-in-a-haystack sequences from TinyStories.

    Haystack: Tokenized real text from roneneldan/TinyStories
    Needle:   Random 8-char alphanumeric string (e.g., X7B9Q2PA)
    Format:
        [story text...] The secret code is X7B9Q2PA. [more story...]
        Question: What is the secret code? Answer: X7B9Q2PA
    """

    def __init__(self, config: SRTTTConfig, tokenizer, stories):
        self.cfg = config
        self.tokenizer = tokenizer
        self.stories = stories
        self.needle_char_len = 8
        # Pre-tokenize fixed template parts
        self._query_prefix = tokenizer.encode(" Question: What is the secret code? Answer:",
                                               add_special_tokens=False)

    def _make_needle(self):
        """Generate random 8-char alphanumeric needle string."""
        return ''.join(random.choices(ALPHANUM, k=self.needle_char_len))

    def generate(self, seq_len: int, needle_depth: float = 0.5,
                 batch_size: int = 1) -> Dict:
        """
        Generate a batch of needle-in-a-haystack sequences.

        Args:
            seq_len: Total sequence length (including needle + query)
            needle_depth: Position of needle as fraction of available space
            batch_size: Number of sequences to generate

        Returns:
            Dict with input_ids [B, seq_len], query_position, needle_tokens, needle_len
        """
        tokenizer = self.tokenizer
        all_input_ids = []
        all_needle_tokens = []

        for _ in range(batch_size):
            # 1. Generate random needle
            needle_str = self._make_needle()

            # 2. Tokenize needle parts
            needle_sentence_ids = tokenizer.encode(
                f" The secret code is {needle_str}.", add_special_tokens=False)
            answer_ids = tokenizer.encode(f" {needle_str}", add_special_tokens=False)
            query_ids = self._query_prefix + answer_ids  # query + answer

            # 3. Calculate how many haystack tokens we need
            overhead = len(needle_sentence_ids) + len(query_ids)
            haystack_needed = seq_len - overhead
            assert haystack_needed > 0, (
                f"seq_len={seq_len} too short for needle({len(needle_sentence_ids)}) "
                f"+ query({len(query_ids)}) = {overhead}")

            # 4. Build haystack from random stories
            haystack_ids = []
            safety = 0
            while len(haystack_ids) < haystack_needed and safety < 200:
                story_idx = random.randint(0, len(self.stories) - 1)
                story_text = self.stories[story_idx]
                story_ids = tokenizer.encode(story_text, add_special_tokens=False)
                haystack_ids.extend(story_ids)
                safety += 1
            haystack_ids = haystack_ids[:haystack_needed]

            # 5. Insert needle at depth
            insert_pos = int(needle_depth * len(haystack_ids))
            insert_pos = max(0, min(insert_pos, len(haystack_ids)))

            seq = (haystack_ids[:insert_pos]
                   + needle_sentence_ids
                   + haystack_ids[insert_pos:]
                   + query_ids)

            # Ensure exact length
            if len(seq) > seq_len:
                seq = seq[:seq_len]
            elif len(seq) < seq_len:
                # Pad with eos tokens
                seq = seq + [tokenizer.eos_token_id] * (seq_len - len(seq))

            all_input_ids.append(torch.tensor(seq, dtype=torch.long))
            all_needle_tokens.append(torch.tensor(answer_ids, dtype=torch.long))

        input_ids = torch.stack(all_input_ids)
        # query_position = start of answer tokens (after query prefix)
        query_position = seq_len - len(answer_ids)
        needle_len = len(answer_ids)

        return {
            "input_ids": input_ids,
            "needle_position": int(needle_depth * haystack_needed),
            "needle_tokens": torch.stack(all_needle_tokens) if batch_size == 1 else all_needle_tokens,
            "query_position": query_position,
            "seq_len": seq_len,
            "needle_depth": needle_depth,
            "needle_len": needle_len,
        }


# --- Data Pipeline Test ---
def test_data_pipeline():
    gen = TinyStoriesNeedleGenerator(CFG, _TOKENIZER, _STORIES_LIST)
    sample = gen.generate(seq_len=1024, needle_depth=0.5, batch_size=1)
    assert sample["input_ids"].shape[0] == 1
    assert sample["input_ids"].shape[1] == 1024
    needle_str = _TOKENIZER.decode(sample["needle_tokens"][0])
    print(f"[OK] TinyStories data pipeline test passed | Needle='{needle_str.strip()}' "
          f"({sample['needle_len']} tokens) at depth={sample['needle_depth']}")

test_data_pipeline()


# %% Cell 8.5: Training Function

def train_srttt(model_obj, config: SRTTTConfig, train_seq_len: int,
                generator,
                n_steps: int = 600, lr: float = 1e-3, log_every: int = 25,
                accum_steps: int = 4, use_cache: bool = False,
                cache_warmup_step: int = None,
                label: str = "SR-TTT"):
    """
    Train the SR-TTT model on needle-in-a-haystack sequences.

    Two-stage training support:
        If cache_warmup_step is set, the model trains with use_cache=False
        for the first (cache_warmup_step - 1) steps, then switches to
        use_cache=True at cache_warmup_step. At the transition, all alpha
        gates are reset to 0.02 so the cache starts with a small but
        directly trainable weight (no sigmoid attenuation).
    """
    model_ref = model_obj.module if isinstance(model_obj, nn.DataParallel) else model_obj
    model_ref.train()
    gen = generator

    optimizer = torch.optim.AdamW(model_ref.parameters(), lr=lr, weight_decay=0.01,
                                  betas=(0.9, 0.95))
    opt_steps = n_steps // accum_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_steps,
                                                            eta_min=lr * 0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)

    current_use_cache = use_cache

    print(f"  [{label}] Training config: {n_steps} steps, seq_len={train_seq_len}, lr={lr}")
    if cache_warmup_step:
        print(f"  [{label}] TWO-STAGE: cache OFF for steps 1-{cache_warmup_step-1}, "
              f"cache ON from step {cache_warmup_step}")
    else:
        print(f"  [{label}] Cache: {'ENABLED' if use_cache else 'DISABLED'} (all steps)")
    print(f"  [{label}] Gradient accumulation: {accum_steps} steps (effective batch={accum_steps})")
    print(f"  [{label}] TinyStories needle-in-a-haystack (8-char alphanumeric)\n")

    loss_history = []
    acc_history = []
    best_loss = float('inf')
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, n_steps + 1):
        # --- Two-stage transition ---
        if cache_warmup_step and step == cache_warmup_step:
            current_use_cache = True
            # Reset alpha gates to 0.05 (strong initial signal for cache)
            with torch.no_grad():
                for block in model_ref.blocks:
                    block.gate_param.fill_(0.05)
                    block.cache.reset()
            alpha_vals = [f"{b.alpha.item():.6f}" for b in model_ref.blocks]
            print(f"\n  *** STAGE 2 ACTIVATED at step {step} ***")
            print(f"  *** Alpha gates reset to: {alpha_vals}")

            # --- Freeze backbone, unfreeze only cache-related params ---
            for param in model_ref.parameters():
                param.requires_grad = False
            for block in model_ref.blocks:
                block.gate_param.requires_grad = True
                for param in block.cache_attn.parameters():
                    param.requires_grad = True

            # Rebuild optimizer with only trainable params
            cur_lr = scheduler.get_last_lr()[0]
            trainable_params = [p for p in model_ref.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=cur_lr, weight_decay=0.01,
                                          betas=(0.9, 0.95))
            remaining_opt_steps = (n_steps - step + 1) // accum_steps
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(remaining_opt_steps, 1), eta_min=lr * 0.1)
            scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
            optimizer.zero_grad(set_to_none=True)

            n_trainable = sum(p.numel() for p in trainable_params)
            print(f"  *** Froze backbone. Trainable params: {n_trainable:,} ***")
            print(f"  *** Cache enabled, training continues for {n_steps - step + 1} steps\n")

        model_ref.reset_caches()

        depth = random.uniform(0.05, 0.95)
        sample = gen.generate(seq_len=train_seq_len, needle_depth=depth, batch_size=1)
        input_ids = sample["input_ids"].to(DEVICE)
        query_pos = sample["query_position"]
        needle_tokens = sample["needle_tokens"][0].to(DEVICE)
        needle_len = sample["needle_len"]

        if config.use_amp:
            with autocast('cuda', dtype=config.torch_dtype):
                all_logits, _ = model_ref(input_ids, start_pos=0, use_cache=current_use_cache)
        else:
            all_logits, _ = model_ref(input_ids, start_pos=0, use_cache=current_use_cache)

        answer_logits = all_logits[0, query_pos:query_pos + needle_len]
        answer_loss = F.cross_entropy(answer_logits, needle_tokens)
        scaled_loss = answer_loss / accum_steps
        scaler.scale(scaled_loss).backward()

        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_ref.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            pred_tokens = answer_logits.argmax(dim=-1)
            correct = (pred_tokens == needle_tokens).float().mean().item()
            exact_match = torch.equal(pred_tokens, needle_tokens)

        loss_history.append(answer_loss.item())
        acc_history.append(correct)

        if answer_loss.item() < best_loss:
            best_loss = answer_loss.item()

        if step % log_every == 0 or step == 1:
            avg_loss = np.mean(loss_history[-log_every:])
            avg_acc = np.mean(acc_history[-log_every:])
            cur_lr = scheduler.get_last_lr()[0]
            cache_fill = np.mean([b.cache.count.item() for b in model_ref.blocks])
            stage = "S2" if current_use_cache and cache_warmup_step else ("ON" if current_use_cache else "S1" if cache_warmup_step else "OFF")
            alpha_avg = np.mean([b.alpha.item() for b in model_ref.blocks])
            print(f"  Step {step:4d}/{n_steps} [{stage}] | loss={avg_loss:.4f} | "
                  f"token_acc={avg_acc:.1%} | exact={exact_match} | "
                  f"cache={cache_fill:.0f} | alpha={alpha_avg:.4f} | lr={cur_lr:.2e}")

        del input_ids, all_logits, answer_loss, scaled_loss
        if torch.cuda.is_available() and step % 50 == 0:
            torch.cuda.empty_cache()

    model_ref.eval()
    final_avg_acc = np.mean(acc_history[-50:]) if len(acc_history) >= 50 else np.mean(acc_history)
    final_alpha = np.mean([b.alpha.item() for b in model_ref.blocks])
    print(f"\n  [{label}] Training complete!")
    print(f"  [{label}] Final avg token accuracy (last 50): {final_avg_acc:.1%}")
    print(f"  [{label}] Best loss: {best_loss:.4f}")
    print(f"  [{label}] Final alpha (avg): {final_alpha:.4f}")

    return loss_history, acc_history


# %% Cell 9: TWO-STAGE TRAINING EXPERIMENT

# --- Shared data generator ---
_GENERATOR = TinyStoriesNeedleGenerator(CFG, _TOKENIZER, _STORIES_LIST)

# --- Determine training steps based on hardware ---
if torch.cuda.is_available():
    N_TRAIN_STEPS = 10000  # Full experiment
    STAGE2_START = 7001    # Cache integration begins at step 7001
else:
    N_TRAIN_STEPS = 200    # Quick smoke test on CPU
    STAGE2_START = 141     # 70% of 200

STAGE1_STEPS = STAGE2_START - 1
STAGE2_STEPS = N_TRAIN_STEPS - STAGE1_STEPS


# =====================================================================
#  MODEL A: Pure TTT Baseline (10K steps, no cache)
# =====================================================================
print(f"\n{'='*60}")
print(f"MODEL A: Pure TTT Baseline -- Training {N_TRAIN_STEPS} steps")
print(f"{'='*60}\n")

model_a, n_params_a = create_model(CFG, "Model A (Pure TTT)")
loss_a, acc_a = train_srttt(
    model_a, CFG, train_seq_len=TRAIN_SEQ_LEN, generator=_GENERATOR,
    n_steps=N_TRAIN_STEPS, use_cache=False, label="Model A"
)

# Plot Model A training curves
fig_a, (ax_a1, ax_a2) = plt.subplots(1, 2, figsize=(14, 5))
fig_a.suptitle("Model A (Pure TTT) -- Training Curves", fontweight="bold")
_w = min(50, len(loss_a))
ax_a1.plot(pd.Series(loss_a).rolling(_w).mean(), color='#E74C3C', linewidth=1.5)
_rand_ceil = np.log(CFG.vocab_size)
ax_a1.axhline(y=_rand_ceil, color='gray', linestyle='--', alpha=0.5, label=f'Random ceiling ln({CFG.vocab_size})={_rand_ceil:.1f}')
ax_a1.set_xlabel("Step"); ax_a1.set_ylabel("Loss"); ax_a1.set_title("Answer Loss"); ax_a1.legend()
ax_a2.plot(pd.Series(acc_a).rolling(_w).mean() * 100, color='#E74C3C', linewidth=1.5)
ax_a2.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Random (25%)')
ax_a2.set_xlabel("Step"); ax_a2.set_ylabel("Token Acc (%)"); ax_a2.set_title("Token Accuracy"); ax_a2.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_a_training.png"), dpi=150, bbox_inches="tight")
print(f"[OK] Model A training curves saved")
plt.close()

if torch.cuda.is_available():
    print(f"[OK] Peak VRAM after Model A: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    torch.cuda.empty_cache()


# =====================================================================
#  MODEL B: Two-Stage (7K base + 3K cache integration)
# =====================================================================
print(f"\n{'='*60}")
print(f"MODEL B: Two-Stage SR-TTT -- {STAGE1_STEPS} base + {STAGE2_STEPS} cache")
print(f"{'='*60}\n")

model_b, n_params_b = create_model(CFG, "Model B (Two-Stage SR-TTT)")
loss_b, acc_b = train_srttt(
    model_b, CFG, train_seq_len=TRAIN_SEQ_LEN, generator=_GENERATOR,
    n_steps=N_TRAIN_STEPS, use_cache=False,
    cache_warmup_step=STAGE2_START, label="Model B"
)

# Plot Model B training curves with stage transition marker
fig_b, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(14, 5))
fig_b.suptitle("Model B (Two-Stage SR-TTT) -- Training Curves", fontweight="bold")
_w = min(50, len(loss_b))
ax_b1.plot(pd.Series(loss_b).rolling(_w).mean(), color='#2ECC71', linewidth=1.5)
ax_b1.axhline(y=_rand_ceil, color='gray', linestyle='--', alpha=0.5, label=f'Random ceiling ln({CFG.vocab_size})')
ax_b1.axvline(x=STAGE2_START, color='#3498DB', linestyle='--', linewidth=2, alpha=0.8, label=f'Stage 2 (step {STAGE2_START})')
ax_b1.set_xlabel("Step"); ax_b1.set_ylabel("Loss"); ax_b1.set_title("Answer Loss"); ax_b1.legend()
ax_b2.plot(pd.Series(acc_b).rolling(_w).mean() * 100, color='#2ECC71', linewidth=1.5)
ax_b2.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Random (25%)')
ax_b2.axvline(x=STAGE2_START, color='#3498DB', linestyle='--', linewidth=2, alpha=0.8, label=f'Stage 2 (step {STAGE2_START})')
ax_b2.set_xlabel("Step"); ax_b2.set_ylabel("Token Acc (%)"); ax_b2.set_title("Token Accuracy"); ax_b2.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_b_training.png"), dpi=150, bbox_inches="tight")
print(f"[OK] Model B training curves saved")
plt.close()

if torch.cuda.is_available():
    print(f"[OK] Peak VRAM after Model B: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")


# %% Cell 10: EVALUATION -- Same Benchmark, Both Models

def evaluate_needle_retrieval(model_obj, config: SRTTTConfig,
                              test_lengths: List[int],
                              generator,
                              use_cache: bool = True,
                              label: str = "Model",
                              n_trials: int = 30) -> pd.DataFrame:
    """Evaluate needle retrieval accuracy across context lengths and depths."""
    model_ref = model_obj.module if isinstance(model_obj, nn.DataParallel) else model_obj
    model_ref.eval()
    gen = generator
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    results = []

    for seq_len in test_lengths:
        for depth in depths:
            n_correct = 0
            token_acc_sum = 0.0
            top5_acc_sum = 0.0
            for trial in range(n_trials):
                model_ref.reset_caches()
                sample = gen.generate(seq_len=seq_len, needle_depth=depth, batch_size=1)
                input_ids = sample["input_ids"].to(DEVICE)
                needle_tokens = sample["needle_tokens"][0]
                query_pos = sample["query_position"]
                needle_len = sample["needle_len"]
                window = config.window_size

                with torch.no_grad():
                    for w_start in range(0, seq_len, window):
                        w_end = min(w_start + window, seq_len)
                        chunk = input_ids[:, w_start:w_end]
                        if config.use_amp:
                            with autocast('cuda', dtype=config.torch_dtype):
                                logits, _ = model_ref(chunk, start_pos=w_start, use_cache=use_cache)
                        else:
                            logits, _ = model_ref(chunk, start_pos=w_start, use_cache=use_cache)

                # Extract prediction from the last window's logits
                local_qpos = query_pos - (seq_len - logits.shape[1])
                if local_qpos < 0 or local_qpos + needle_len > logits.shape[1]:
                    continue

                pred_logits = logits[0, local_qpos:local_qpos + needle_len]
                pred_tokens = pred_logits.argmax(dim=-1).cpu()
                target = needle_tokens

                if torch.equal(pred_tokens, target):
                    n_correct += 1
                token_acc_sum += (pred_tokens == target).float().mean().item()
                top5 = pred_logits.topk(5, dim=-1).indices.cpu()
                top5_acc_sum += (top5 == target.unsqueeze(-1)).any(dim=-1).float().mean().item()
                del input_ids, logits

            accuracy = n_correct / n_trials
            avg_token_acc = token_acc_sum / n_trials
            avg_top5_acc = top5_acc_sum / n_trials
            results.append({
                "model": label, "length": seq_len, "depth": depth,
                "accuracy": accuracy, "token_acc": avg_token_acc, "top5_acc": avg_top5_acc,
                "n_correct": n_correct, "n_total": n_trials,
            })
            print(f"  [{label}] len={seq_len:>6} depth={depth:.2f} -> "
                  f"exact={accuracy:.0%} token_acc={avg_token_acc:.1%} top5={avg_top5_acc:.1%}")

    return pd.DataFrame(results)


# =====================================================================
#  EVALUATE MODEL A (Pure TTT, cache disabled)
# =====================================================================
print(f"\n{'='*60}")
print("EVALUATING Model A: Pure TTT (Cache Disabled)")
print(f"{'='*60}\n")

model_a_ref = model_a.module if isinstance(model_a, nn.DataParallel) else model_a
eval_a_df = evaluate_needle_retrieval(model_a, CFG, TEST_LENGTHS, generator=_GENERATOR,
                                       use_cache=False, label="Model A (Pure TTT)")

a_exact = eval_a_df["accuracy"].mean()
a_token = eval_a_df["token_acc"].mean()
a_top5 = eval_a_df["top5_acc"].mean()
print(f"\n[Model A] Exact: {a_exact:.1%} | Token: {a_token:.1%} | Top-5: {a_top5:.1%}")


# =====================================================================
#  EVALUATE MODEL B (Two-Stage, cache enabled at eval)
# =====================================================================
print(f"\n{'='*60}")
print("EVALUATING Model B: Two-Stage SR-TTT (Cache Enabled)")
print(f"{'='*60}\n")

model_b_ref = model_b.module if isinstance(model_b, nn.DataParallel) else model_b
eval_b_df = evaluate_needle_retrieval(model_b, CFG, TEST_LENGTHS, generator=_GENERATOR,
                                       use_cache=True, label="Model B (Two-Stage SR-TTT)")

b_exact = eval_b_df["accuracy"].mean()
b_token = eval_b_df["token_acc"].mean()
b_top5 = eval_b_df["top5_acc"].mean()
print(f"\n[Model B] Exact: {b_exact:.1%} | Token: {b_token:.1%} | Top-5: {b_top5:.1%}")

# Show final alpha gate values for Model B
print(f"\nModel B final alpha gates:")
for i, block in enumerate(model_b_ref.blocks):
    print(f"  Layer {i}: alpha={block.alpha.item():.6f} (gate_param={block.gate_param.item():.4f})")

print("\nCache statistics per layer (Model B):")
for i, stats in enumerate(model_b_ref.cache_stats()):
    print(f"  Layer {i:2d}: {stats}")


# =====================================================================
#  THE SHOWDOWN
# =====================================================================
exact_delta = b_exact - a_exact
token_delta = b_token - a_token
top5_delta = b_top5 - a_top5

print(f"\n{'='*60}")
print(f"SHOWDOWN: Pure TTT vs Two-Stage SR-TTT")
print(f"{'='*60}")
print(f"  Model A (Pure TTT, 10K):       exact={a_exact:.1%} | token_acc={a_token:.1%} | top5={a_top5:.1%}")
print(f"  Model B (Two-Stage, 7K+3K):    exact={b_exact:.1%} | token_acc={b_token:.1%} | top5={b_top5:.1%}")
print(f"  -----------------------------------------------")
print(f"  Delta:                          exact={exact_delta:+.1%} | token_acc={token_delta:+.1%} | top5={top5_delta:+.1%}")
print(f"{'='*60}")

# Per-depth comparison
print(f"\n{'='*60}")
print("PER-DEPTH COMPARISON (Exact Match %)")
print(f"{'='*60}")
for seq_len in TEST_LENGTHS:
    print(f"\n  Context Length = {seq_len}")
    print(f"  {'Depth':>8} | {'Model A':>10} | {'Model B':>10} | {'Delta':>10}")
    print(f"  {'-'*45}")
    for depth in [0.1, 0.25, 0.5, 0.75, 0.9]:
        a_val = eval_a_df[(eval_a_df["length"] == seq_len) & (eval_a_df["depth"] == depth)]["accuracy"].values[0]
        b_val = eval_b_df[(eval_b_df["length"] == seq_len) & (eval_b_df["depth"] == depth)]["accuracy"].values[0]
        delta = b_val - a_val
        marker = " ***" if delta > 0.05 else ""
        print(f"  {depth:>8.2f} | {a_val:>9.0%} | {b_val:>9.0%} | {delta:>+9.0%}{marker}")


# %% Cell 11: Visualization

sns.set_theme(style="darkgrid", palette="deep", font_scale=1.1)
combined_df = pd.concat([eval_a_df, eval_b_df], ignore_index=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Two-Stage Experiment: Pure TTT vs SR-TTT (7K base + 3K cache)\n",
             fontsize=16, fontweight="bold", y=0.98)

# Plot 1: Model A heatmap
a_pivot = eval_a_df.pivot(index="depth", columns="length", values="token_acc")
ax1 = axes[0, 0]
sns.heatmap(a_pivot, annot=True, fmt=".1%", cmap="YlOrRd_r", vmin=0, vmax=1, ax=ax1,
            cbar_kws={"label": "Token Accuracy"})
ax1.set_title("Model A (Pure TTT) -- Token Accuracy", fontweight="bold")
ax1.set_xlabel("Context Length"); ax1.set_ylabel("Needle Depth")

# Plot 2: Model B heatmap
b_pivot = eval_b_df.pivot(index="depth", columns="length", values="token_acc")
ax2 = axes[0, 1]
sns.heatmap(b_pivot, annot=True, fmt=".1%", cmap="YlGn", vmin=0, vmax=1, ax=ax2,
            cbar_kws={"label": "Token Accuracy"})
ax2.set_title("Model B (Two-Stage SR-TTT) -- Token Accuracy", fontweight="bold")
ax2.set_xlabel("Context Length"); ax2.set_ylabel("Needle Depth")

# Plot 3: Difference heatmap (B - A)
diff_df = eval_b_df.copy()
diff_df["delta"] = eval_b_df["token_acc"].values - eval_a_df["token_acc"].values
diff_pivot = diff_df.pivot(index="depth", columns="length", values="delta")
ax3 = axes[1, 0]
max_abs = max(abs(diff_pivot.min().min()), abs(diff_pivot.max().max()), 0.01)
sns.heatmap(diff_pivot, annot=True, fmt="+.1%", cmap="RdYlGn", center=0,
            vmin=-max_abs, vmax=max_abs, ax=ax3, cbar_kws={"label": "Delta Token Accuracy"})
ax3.set_title("SR-TTT Advantage (Model B - Model A)", fontweight="bold")
ax3.set_xlabel("Context Length"); ax3.set_ylabel("Needle Depth")

# Plot 4: Grouped bar chart by depth
ax4 = axes[1, 1]
depth_a = eval_a_df.groupby("depth")["accuracy"].mean()
depth_b = eval_b_df.groupby("depth")["accuracy"].mean()
xd = np.arange(len(depth_a))
wd = 0.35
bars_a = ax4.bar(xd - wd/2, depth_a.values * 100, wd, label="Model A (Pure TTT)",
                 color="#E74C3C", edgecolor="black", linewidth=0.5)
bars_b = ax4.bar(xd + wd/2, depth_b.values * 100, wd, label="Model B (Two-Stage SR-TTT)",
                 color="#2ECC71", edgecolor="black", linewidth=0.5)
ax4.set_xlabel("Needle Depth"); ax4.set_ylabel("Exact Match (%)")
ax4.set_title("Exact Match by Needle Depth (All Lengths)", fontweight="bold")
ax4.set_xticks(xd); ax4.set_xticklabels([f"{d:.2f}" for d in depth_a.index])
ax4.legend(); ax4.set_ylim(0, 105)
for bar in bars_a:
    ax4.annotate(f'{bar.get_height():.0f}%', (bar.get_x() + bar.get_width()/2., bar.get_height()),
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars_b:
    ax4.annotate(f'{bar.get_height():.0f}%', (bar.get_x() + bar.get_width()/2., bar.get_height()),
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "ab_experiment_results.png"), dpi=150, bbox_inches="tight")
print(f"\n[OK] A/B experiment plot saved")
plt.close()

# Training comparison plot with stage transition marker
fig_train, axes_t = plt.subplots(1, 2, figsize=(14, 5))
fig_train.suptitle("Training Curves: Pure TTT vs Two-Stage SR-TTT", fontweight="bold")
_w = min(50, len(loss_a))
axes_t[0].plot(pd.Series(loss_a).rolling(_w).mean(), color='#E74C3C', linewidth=1.5, label="Model A (Pure TTT)")
axes_t[0].plot(pd.Series(loss_b).rolling(_w).mean(), color='#2ECC71', linewidth=1.5, label="Model B (Two-Stage)")
axes_t[0].axhline(y=_rand_ceil, color='gray', linestyle='--', alpha=0.5, label=f'Random ln({CFG.vocab_size})')
axes_t[0].axvline(x=STAGE2_START, color='#3498DB', linestyle=':', linewidth=2, alpha=0.8, label=f'Stage 2 start')
axes_t[0].set_xlabel("Step"); axes_t[0].set_ylabel("Loss"); axes_t[0].set_title("Answer Loss"); axes_t[0].legend()
axes_t[1].plot(pd.Series(acc_a).rolling(_w).mean() * 100, color='#E74C3C', linewidth=1.5, label="Model A (Pure TTT)")
axes_t[1].plot(pd.Series(acc_b).rolling(_w).mean() * 100, color='#2ECC71', linewidth=1.5, label="Model B (Two-Stage)")
axes_t[1].axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Random (25%)')
axes_t[1].axvline(x=STAGE2_START, color='#3498DB', linestyle=':', linewidth=2, alpha=0.8, label=f'Stage 2 start')
axes_t[1].set_xlabel("Step"); axes_t[1].set_ylabel("Token Acc (%)"); axes_t[1].set_title("Token Accuracy"); axes_t[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ab_training_comparison.png"), dpi=150, bbox_inches="tight")
print(f"[OK] Training comparison plot saved")
plt.close()

# Cache utilization plot (Model B only)
fig2, ax_cache = plt.subplots(figsize=(10, 5))
cache_data = model_b_ref.cache_stats()
layers = list(range(len(cache_data)))
fills = [s["fill"] for s in cache_data]
inserts = [s["total_inserts"] for s in cache_data]
evictions = [s["total_evictions"] for s in cache_data]
xc = np.arange(len(layers))
wc = 0.25
ax_cache.bar(xc - wc, fills, wc, label="Current Fill", color="#3498DB")
ax_cache.bar(xc, inserts, wc, label="Total Inserts", color="#2ECC71")
ax_cache.bar(xc + wc, evictions, wc, label="Total Evictions", color="#E74C3C")
ax_cache.set_xlabel("Layer Index"); ax_cache.set_ylabel("Count")
ax_cache.set_title("Model B (Two-Stage SR-TTT) Cache Utilization per Layer", fontweight="bold")
ax_cache.set_xticks(xc); ax_cache.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "srttt_cache_utilization.png"), dpi=150, bbox_inches="tight")
print(f"[OK] Cache utilization plot saved")
plt.close()


# %% Cell 12: Export Weights & Results

print(f"\n{'='*60}")
print("EXPORTING RESULTS")
print(f"{'='*60}\n")

for name, m_ref in [("model_a_pure_ttt.pt", model_a_ref), ("model_b_twostage_srttt.pt", model_b_ref)]:
    wpath = os.path.join(OUTPUT_DIR, name)
    torch.save({
        "model_state_dict": m_ref.state_dict(),
        "config": {"n_layers": CFG.n_layers, "d_model": CFG.d_model,
                    "n_heads": CFG.n_heads, "d_head": CFG.d_head,
                    "d_ff": CFG.d_ff, "cache_cap": CFG.cache_cap,
                    "vocab_size": CFG.vocab_size},
    }, wpath)
    print(f"[OK] {name} saved ({os.path.getsize(wpath) / 1e6:.1f} MB)")

results_path = os.path.join(OUTPUT_DIR, "ab_benchmark_results.csv")
combined_df.to_csv(results_path, index=False)
print(f"[OK] Benchmark results saved: {results_path}")

# Final summary
print(f"\n{'='*60}")
print("FINAL SUMMARY -- TWO-STAGE EXPERIMENT")
print(f"{'='*60}")
print(f"  Architecture:   {CFG.n_layers}L / d={CFG.d_model} / {CFG.n_heads}H")
print(f"  Parameters:     {n_params_a / 1e6:.1f}M (each model)")
print(f"  Cache:          cap={CFG.cache_cap}, window={CFG.window_size}")
print(f"  GPU:            {GPU_NAME} x {GPU_COUNT}")
if torch.cuda.is_available():
    print(f"  Peak VRAM:      {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
print(f"  Training A:     {N_TRAIN_STEPS} steps pure TTT")
print(f"  Training B:     {STAGE1_STEPS} base + {STAGE2_STEPS} cache (warmup at step {STAGE2_START})")
print(f"  Eval lengths:   {TEST_LENGTHS}")
print(f"")
print(f"  Model A (Pure TTT, 10K):       exact={a_exact:.1%} | token_acc={a_token:.1%}")
print(f"  Model B (Two-Stage, 7K+3K):    exact={b_exact:.1%} | token_acc={b_token:.1%}")
print(f"  -----------------------------------------------")
print(f"  Delta:                          exact={exact_delta:+.1%} | token_acc={token_delta:+.1%}")
if exact_delta > 0:
    print(f"\n  *** TWO-STAGE SR-TTT WINS: +{exact_delta:.1%} exact match improvement ***")
elif exact_delta == 0:
    print(f"\n  Result: Tie")
else:
    print(f"\n  Result: Pure TTT baseline wins by {-exact_delta:.1%}")
print(f"\n  Outputs saved to: {OUTPUT_DIR}")
print(f"{'='*60}")
print("\n[OK] Two-stage experiment complete!")

