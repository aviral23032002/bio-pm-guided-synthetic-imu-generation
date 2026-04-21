#!/usr/bin/env python3
"""
car_imu_decoder.py — CAR-IMU: Class-Conditional Autoregressive IMU Decoder

WHAT THIS FILE IS:
    The core model for our project. It is a small GPT-style transformer that
    learns to predict Bio-PM token sequences one step at a time, conditioned
    on an activity class label.

HOW IT FITS INTO THE PIPELINE:
    Bio-PM encoder (frozen) → tokens Z (192×64) per window
                                        ↓
                             [CLS_c] prepended to Z
                                        ↓
                             CAR-IMU Decoder (this file, trained)
                             predicts Z step by step
                                        ↓
                             At inference: sample NEW Z sequences
                             for minority activity classes
                                        ↓
                             Augment real Z pool → better HAR classifier

ARCHITECTURE:
    - 4-layer causal transformer (like GPT, not BERT — generates left-to-right)
    - D = 64  (matches Bio-PM token dimension exactly)
    - 4 attention heads, FFN dim = 128
    - Activity-class prefix token [CLS_c] learned embedding
    - Output head: Linear(64 → 64), no activation
    - ~1.2M parameters — trains in minutes on CPU

DESIGN DECISION — NO SUBJECT CONDITIONING:
    Week 1 analysis showed 0% subject separation in Bio-PM's embedding space.
    Mean-pooled tokens are nearly identical across all 36 subjects (norms 5.5–6.7).
    So we use ACTIVITY-ONLY conditioning — simpler and just as effective.
    This is a paper contribution: we empirically ablated subject conditioning.

TRAINING (done in train_car_imu.py):
    Input:  [CLS_c] + z_1 + z_2 + ... + z_{L-1}   (L=192 tokens, shifted right)
    Target:           z_1 + z_2 + ... + z_L
    Loss:   MSE on every token position (teacher forcing)
    Why MSE? We want to stay close to the real Bio-PM embedding space geometry.

SAMPLING (done in generate_synthetic.py):
    1. Start with just [CLS_c]
    2. Predict z_1, add small Gaussian noise (temperature τ)
    3. Feed [CLS_c, z_1] → predict z_2
    4. Repeat until we have L=192 tokens
    5. This synthetic token sequence Ẑ can be used directly as augmentation data
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constants matching Bio-PM token space ─────────────────────────────────────
TOKEN_DIM    = 64    # D: Bio-PM encoder output dimension per token
SEQ_LEN      = 192   # L: number of tokens per window (from preprocessing)
NUM_CLASSES  = 6     # WISDM: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing
ACTIVITY_NAMES = {
    0: "Walking", 1: "Jogging", 2: "Upstairs",
    3: "Downstairs", 4: "Sitting", 5: "Standing"
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  POSITIONAL ENCODING
#     Adds position information to each token so the model knows "this is
#     token #5 out of 192". Without this, the transformer treats all positions
#     identically.
# ══════════════════════════════════════════════════════════════════════════════
class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal position embeddings (no learnable params).
    Same as the original "Attention Is All You Need" paper.

    For each position pos and each dimension i:
        PE(pos, 2i)   = sin(pos / 10000^(2i/D))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the fixed PE table: shape (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)              # (1, max_len, d_model)
        self.register_buffer("pe", pe)   # not a parameter, but saved with model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) — adds positional encoding to each token."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CAUSAL SELF-ATTENTION
#     "Causal" means each token can only look at itself and tokens BEFORE it.
#     This is what makes generation work — at step t, the model only sees z_1..z_t
#     and cannot cheat by looking at future tokens.
# ══════════════════════════════════════════════════════════════════════════════
class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal (upper-triangular) mask.

    Standard scaled dot-product attention:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    The causal mask fills future positions with -inf so softmax gives them 0 weight.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5     # 1/sqrt(d_k) scaling factor

        # Single linear projects input into Q, K, V all at once
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)  where L = sequence length (includes [CLS_c] prefix)
        Returns: (B, L, D)
        """
        B, L, D = x.shape

        # Project to Q, K, V and split into heads
        qkv = self.qkv_proj(x)                         # (B, L, 3D)
        q, k, v = qkv.split(D, dim=-1)                 # each: (B, L, D)

        # Reshape for multi-head: (B, n_heads, L, head_dim)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: (B, n_heads, L, L)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # ── Causal mask: token at position i cannot attend to position j > i ──
        # Creates upper triangle of -inf so those positions get ~0 weight in softmax
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values: (B, n_heads, L, head_dim) → (B, L, D)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TRANSFORMER DECODER BLOCK
#     One layer = LayerNorm → CausalAttention → Residual → LayerNorm → FFN → Residual
#     We stack 4 of these.
# ══════════════════════════════════════════════════════════════════════════════
class TransformerDecoderBlock(nn.Module):
    """
    Pre-LN transformer block (LayerNorm BEFORE attention, not after).
    Pre-LN is more stable to train than the original post-LN design.

    FFN is a simple 2-layer MLP: D → 2D → D with GELU activation.
    """
    def __init__(self, d_model: int, n_heads: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),                    # GELU is smoother than ReLU, standard in GPT
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer with residual connection
        x = x + self.attn(self.ln1(x))
        # FFN sub-layer with residual connection
        x = x + self.ffn(self.ln2(x))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CAR-IMU DECODER  (the full model)
# ══════════════════════════════════════════════════════════════════════════════
class CARIMUDecoder(nn.Module):
    """
    Class-conditional Autoregressive IMU Decoder.

    At training time (teacher forcing):
        Input:  [CLS_c]  z_1  z_2  ...  z_{L-1}    (length = L+1)
        Output: (ignored) z_1  z_2  ...  z_L         (shift by 1)
        Loss:   MSE between output[1:] and target tokens[1:]

    At inference time (autoregressive):
        Step 0: input = [CLS_c]           → predict ẑ_1
        Step 1: input = [CLS_c, ẑ_1]     → predict ẑ_2
        ...
        Step L: input = [CLS_c, ẑ_1..L-1] → predict ẑ_L

    Args:
        num_classes:  number of activity classes (6 for WISDM)
        d_model:      token embedding dimension (64, matches Bio-PM)
        n_layers:     number of transformer blocks (4)
        n_heads:      number of attention heads (4)
        ffn_dim:      feedforward hidden dim (128 = 2×D)
        dropout:      dropout rate
        max_seq_len:  maximum sequence length including [CLS_c] prefix
    """

    def __init__(
        self,
        num_classes: int  = NUM_CLASSES,
        d_model:     int  = TOKEN_DIM,
        n_layers:    int  = 4,
        n_heads:     int  = 4,
        ffn_dim:     int  = 128,          # 2 × D
        dropout:     float = 0.1,
        max_seq_len: int  = SEQ_LEN + 1,  # +1 for [CLS_c]
    ):
        super().__init__()
        self.d_model      = d_model
        self.seq_len      = SEQ_LEN
        self.num_classes  = num_classes

        # ── [CLS_c]: one learned embedding vector per activity class ────────
        # When we want to generate "Jogging", we look up class_emb[1]
        # This single vector is the ONLY conditioning signal (no subject vector)
        self.class_emb = nn.Embedding(num_classes, d_model)
        nn.init.trunc_normal_(self.class_emb.weight, std=0.02)

        # ── Positional encoding (sinusoidal, fixed) ─────────────────────────
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=max_seq_len, dropout=dropout)

        # ── Input projection: Bio-PM tokens are already D-dim, but we add
        #    a learned linear to let the model scale/rotate the input space
        self.input_proj = nn.Linear(d_model, d_model, bias=False)

        # ── 4 causal transformer blocks ─────────────────────────────────────
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        # ── Final layer norm + output head ──────────────────────────────────
        # Output head maps hidden state back to token space (D → D)
        # No activation — we want real-valued embeddings, not probabilities
        self.ln_final  = nn.LayerNorm(d_model)
        self.out_head  = nn.Linear(d_model, d_model, bias=False)

        # ── Weight tying: tie output head to input projection ───────────────
        # Common trick — helps regularization. Means the model learns
        # "tokens that go in" and "tokens that come out" in the same space.
        self.out_head.weight = self.input_proj.weight

        self._init_weights()

    def _init_weights(self):
        """Small initialisation for stable early training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ──────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        tokens: torch.Tensor,      # (B, L, 64) — real Bio-PM tokens from token_store
        class_labels: torch.Tensor # (B,)        — integer activity class 0-5
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass for training.

        Takes the full token sequence and prepends [CLS_c].
        Returns predicted tokens at every position.

        Args:
            tokens:       (B, L, 64)  real token sequences from token_store.hdf5
            class_labels: (B,)        integer labels (0=Walking ... 5=Standing)

        Returns:
            logits: (B, L+1, 64)  predicted token at each position
                    → during training we compare logits[:, 1:, :] to tokens
                       (shift by 1: position 0 is [CLS_c], not a real token)
        """
        B, L, D = tokens.shape

        # 1. Look up class embedding: (B, 1, D)
        cls_token = self.class_emb(class_labels).unsqueeze(1)  # (B, 1, 64)

        # 2. Project real tokens into model space: (B, L, D)
        tok_proj = self.input_proj(tokens)

        # 3. Concatenate [CLS_c] + tokens → input sequence of length L+1
        #    Shape: (B, L+1, 64)
        seq = torch.cat([cls_token, tok_proj], dim=1)

        # 4. Add positional encoding
        seq = self.pos_enc(seq)

        # 5. Pass through 4 causal transformer layers
        for layer in self.layers:
            seq = layer(seq)

        # 6. Final layer norm + output projection → (B, L+1, 64)
        seq = self.ln_final(seq)
        logits = self.out_head(seq)

        return logits   # training uses logits[:, 1:, :] vs tokens as target

    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(
        self,
        class_label:  int,
        n_tokens:     int   = SEQ_LEN,
        temperature:  float = 1.0,
        device:       str   = "cpu",
    ) -> torch.Tensor:
        """
        Autoregressive sampling — generates ONE synthetic token sequence.

        At each step, the model predicts the next token given everything before.
        We add Gaussian noise scaled by `temperature` to get variation.
            τ = 0.0  →  greedy (always use the predicted mean, no randomness)
            τ = 0.5  →  mild stochastic  ← good starting point
            τ = 1.0  →  full noise

        Args:
            class_label: which activity to generate (0=Walking ... 5=Standing)
            n_tokens:    how many tokens to generate (default 192 = SEQ_LEN)
            temperature: noise level for stochastic sampling
            device:      'cpu' or 'cuda'

        Returns:
            generated: (n_tokens, 64)  synthetic token sequence Ẑ
        """
        self.eval()

        # Start with just the class token: shape (1, 1, 64)
        label_tensor = torch.tensor([class_label], dtype=torch.long, device=device)
        context = self.class_emb(label_tensor).unsqueeze(1)  # (1, 1, 64)

        generated_tokens = []

        for step in range(n_tokens):
            # Add positional encoding to current context
            context_pe = self.pos_enc(context)

            # Pass through all transformer layers
            h = context_pe
            for layer in self.layers:
                h = layer(h)

            # Get prediction for the LAST position only
            h = self.ln_final(h)
            next_token_pred = self.out_head(h[:, -1, :])  # (1, 64)

            # Add temperature-scaled Gaussian noise for stochastic sampling
            if temperature > 0.0:
                noise = torch.randn_like(next_token_pred) * temperature
                next_token_pred = next_token_pred + noise

            generated_tokens.append(next_token_pred)  # save this token

            # Append predicted token to context for next step
            # Must project it back through input_proj to stay in model space
            next_proj = self.input_proj(next_token_pred).unsqueeze(1)  # (1, 1, 64)
            context = torch.cat([context, next_proj], dim=1)           # grow by 1

        # Stack all generated tokens: (n_tokens, 64)
        return torch.cat(generated_tokens, dim=0).squeeze()

    @torch.no_grad()
    def sample_batch(
        self,
        class_label: int,
        n_samples:   int,
        n_tokens:    int   = SEQ_LEN,
        temperature: float = 1.0,
        device:      str   = "cpu",
    ) -> torch.Tensor:
        """
        Generate multiple synthetic sequences for one class.

        Args:
            class_label: activity class to generate
            n_samples:   how many sequences to generate
            n_tokens:    tokens per sequence (192)
            temperature: sampling noise

        Returns:
            (n_samples, n_tokens, 64)  batch of synthetic token sequences
        """
        samples = []
        for _ in range(n_samples):
            seq = self.sample(class_label, n_tokens, temperature, device)
            samples.append(seq.unsqueeze(0))
        return torch.cat(samples, dim=0)

    def count_parameters(self) -> int:
        """Returns number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# Run this file directly to verify the model builds and shapes are correct:
#     python car_imu_decoder.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("CAR-IMU Decoder — Architecture Sanity Check")
    print("=" * 60)

    model = CARIMUDecoder(
        num_classes = NUM_CLASSES,
        d_model     = TOKEN_DIM,     # 64
        n_layers    = 4,
        n_heads     = 4,
        ffn_dim     = 128,           # 2 × 64
        dropout     = 0.1,
    )
    model.eval()

    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"  (should be ~1-2M for a lightweight decoder)")

    # ── Test forward pass (training mode) ─────────────────────────────────────
    B, L, D = 4, SEQ_LEN, TOKEN_DIM   # batch=4, 192 tokens, 64 dims
    fake_tokens = torch.randn(B, L, D)
    fake_labels = torch.randint(0, NUM_CLASSES, (B,))

    logits = model(fake_tokens, fake_labels)
    print(f"\nForward pass (teacher forcing):")
    print(f"  Input tokens: {fake_tokens.shape}")
    print(f"  Labels:       {fake_labels.tolist()}")
    print(f"  Output shape: {logits.shape}  (B, L+1, D)")

    # During training we compute loss on logits[:, 1:, :] vs fake_tokens
    pred = logits[:, 1:, :]   # shift: drop position 0 ([CLS_c] prediction)
    loss = F.mse_loss(pred, fake_tokens)
    print(f"  MSE loss (random init, should be ~1.0): {loss.item():.4f}")

    # ── Test autoregressive sampling ───────────────────────────────────────────
    print(f"\nAutoregressive sampling:")
    for cls_id, cls_name in ACTIVITY_NAMES.items():
        synth = model.sample(cls_id, n_tokens=SEQ_LEN, temperature=0.5)
        print(f"  Class {cls_id} [{cls_name:<12}] → shape {synth.shape}  "
              f"mean={synth.mean():.3f}  std={synth.std():.3f}")

    print(f"\n✅ All checks passed!")
    print(f"\nNext step: run  python train_car_imu.py")
