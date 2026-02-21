# SR-TTT: Surprisal-Aware Residual Test-Time Training

## Abstract / TL;DR
Test-Time Training (TTT) language models replace the standard KV-cache with a set of hidden state "fast weights" ($W_{fast}$) updated via self-supervised learning during inference. While this theoretically allows for infinite context windows with $O(1)$ memory footprint, pure TTT architectures suffer from a catastrophic failure in **exact-recall** tasks (like Needle-in-a-Haystack). Because the fast weights aggressively compress the context into an information bottleneck, highly surprising or unique tokens (like an alphanumeric needle) are "forgotten" as consecutive gradient updates overwrite them.

**SR-TTT** solves this by augmenting the TTT backbone with a loss-gated sparse memory mechanism: the **Residual Cache**. By dynamically routing only incompressible, "surprising" tokens to a traditional exact-attention cache and fusing the outputs, SR-TTT achieves the best of both worlds: $O(1)$ memory for background context, and standard $N^2$ exact attention solely for critical needles.

## The Architecture
SR-TTT maintains the infinite-context compression of standard Test-Time Training while bolting on a parallel memory track:

1. **The Surprisal Filter**: During the TTT inner-loop forward pass, we calculate the per-token reconstruction loss ($||z - v||^2$). Tokens that exceed a dynamically smoothed threshold (using Exponential Moving Average) and have high chunk-level loss are flagged as "surprising" (i.e., highly incompressible). By training on a very low-entropy, structured background dataset (TinyStories), unique high-entropy needles reliably spike the surprisal loss.
2. **The Residual Cache**: Flagged tokens (their post-RoPE Keys and Values) are routed to a sparse, fixed-capacity memory bank.
3. **Gated Cache Attention ($\alpha$)**: A multi-head attention module queries the Residual cache. The output is fused back into the main TTT stream via a learned gate vector ($\alpha$). 

$$ \text{Output} = \text{TTT}(x) + \alpha \cdot \text{CacheAttention}(x) $$

## The Training Curriculum
Training SR-TTT directly from scratch end-to-end fails due to **Cold Start Noise**. In the early stages of training, the TTT backbone essentially produces random noise. Because the network aggressively attempts to minimize loss early on, it learns to completely shut off the noisy, uncalibrated cache by forcing the $\alpha$ gates to $0.0$, effectively reverting to a pure TTT model and ignoring the cache entirely.

To solve this, we implemented a **Two-Stage Training Curriculum**:
* **Stage 1 (Steps 1 to 7,000): Base TTT Training.** The cache is disabled, and the model focuses entirely on learning the underlying language modeling task and updating its fast weights.
* **Stage 2 (Steps 7,001 to 10,000): Cache Warmup & Integration.** We freeze the TTT backbone parameters, enable the cache, and initialize the $\alpha$ gates. By freezing everything except the sparse memory gates, the network is *forced* to route gradients through the $\alpha$ module to improve its loss, gradually pulling the gates open to ~10% ($\alpha \approx 0.085$) and learning to utilize the Residual Cache effectively.

## Results
We evaluated SR-TTT comprehensively using an 8-character alphanumeric Needle-in-a-Haystack protocol against a pure TTT Baseline model. Both architectures were trained for 10,000 steps ($15.8\text{M}$ parameters, evaluated on sequence lengths up to 4096).

At the **2048-token context length**, SR-TTT demonstrated massive exact-match improvements in mid-sequence retrieval, effectively rescuing needles that pure TTT's sliding window explicitly "forgotten":

| Depth | Pure TTT (Baseline) | Two-Stage SR-TTT | Delta |
| :--- | :--- | :--- | :--- |
| **0.10** | 30% | 33% | +3% |
| **0.25** | 37% | 37% | +0% |
| **0.50** | 10% | 33% | **+23%** |
| **0.75** | 17% | 37% | **+20%** |
| **0.90** | 23% | 23% | +0% |

**Overall Average Exact Match (All Depths)**:  `10.9%` (SR-TTT) vs `7.8%` (Pure TTT) $\rightarrow$ **+3.1% Win**

The validation logs confirm the core hypothesis: the Surprisal Filter successfully identified the needle, parked it in the Residual Cache, and the learned fusion gate properly retrieved it when the standard TTT fast weights lost it.

## Known Limitations & Future Work
While the 2048-token baseline victories validate the core sparse memory routing, there are architectural limits intentionally left out of scope for this Proof of Concept:

* **The RoPE Scaling Wall (4096 length failure):** Both architectures collapsed completely (scoring $0\%$ exact match) at the 4096-token evaluation length. Because the model was explicitly trained at `seq_len = 2048`, the standard Rotary Position Embeddings (RoPE) suffer a known, catastrophic zero-shot extrapolation failure at unseen positions. Implementing Dynamic NTK / YaRN interpolation or explicitly scaling the training sequence length is required to push the context length boundary further.
* **Eviction Policy Limits:** While the current codebase implements a priority-based eviction mechanism ($\frac{\text{Surprisal}}{1 + \text{Age}}$), it mathematically converges to the same limits as standard FIFO eviction policies when the sequence drastically exceeds the cache capacity limit. At extremely long context lengths, a small cache is guaranteed to evict needles due to raw volume. Future iterations must investigate hierarchical routing, secondary compression, or more aggressive noise-filtering to sustain larger scales.
