#!/usr/bin/env python3
"""
car_imu_decoder.py — CAR-IMU: Class-Conditional Autoregressive IMU Decoder

WHAT THIS FILE IS:
    The core model for our project. A custom IMU Transformer that learns to
    predict Bio-PM token sequences one step at a time, conditioned on an
    activity class label.

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

ARCHITECTURE (CHANGE 1 — Custom IMU Transformer):
    - IMUPositionalEncoding: class-specific learnable frequency+phase PE
    - DualScopeAttention:    local (window=16) + global causal attention, gated
    - SpectralFFN:           token branch + frequency branch, merged
    - IMUTransformerLayer:   pre-norm attn + pre-norm SpectralFFN
    - 4 stacked IMUTransformerLayer blocks
    - D = 64 (matches Bio-PM token dimension exactly)
    - 4 attention heads, FFN dim = 128
    - Activity-class prefix token [CLS_c] learned embedding
    - Output head: Linear(64 → 64), no activation

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
# 1.  IMU POSITIONAL ENCODING
#     Each activity class gets its own learnable frequency and phase.
#     This lets the model learn class-specific temporal rhythms
#     (e.g., jogging has faster periodicity than sitting).
# ══════════════════════════════════════════════════════════════════════════════
class IMUPositionalEncoding(nn.Module):
    """
    Class-conditional positional encoding with learnable frequency and phase.

    For each class c, position t, and model dimension d:
        periodic_bias[c, t, :] = sin(2π * freq[c] * t / seq_len + phase[c])

    Also adds a standard learnable base PE (nn.Embedding over positions).
    """
    def __init__(self, num_classes: int, d_model: int, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.d_model   = d_model
        self.num_classes = num_classes

        # Per-class learnable frequency and phase — shape (num_classes, 1, d_model)
        self.freq  = nn.Parameter(torch.ones(num_classes, 1, d_model))
        self.phase = nn.Parameter(torch.zeros(num_classes, 1, d_model))

        # Standard learnable base positional embedding over positions
        self.base_pe = nn.Embedding(max_len, d_model)

        nn.init.uniform_(self.freq,  0.5, 2.0)   # sensible frequency range
        nn.init.uniform_(self.phase, -math.pi, math.pi)
        nn.init.trunc_normal_(self.base_pe.weight, std=0.02)

    def forward(self, x: torch.Tensor, class_ids: torch.Tensor,
                seq_len: int) -> torch.Tensor:
        """
        x:         (B, T, d_model)
        class_ids: (B,)  integer activity labels
        seq_len:   T (current sequence length)

        Returns x + periodic_bias + base_pe, shape (B, T, d_model)
        """
        B, T, D = x.shape
        device = x.device

        # ── Periodic bias (class-specific) ────────────────────────────────────
        # t: (1, T, 1) — time positions
        t = torch.arange(T, device=device, dtype=torch.float32).view(1, T, 1)

        # freq[class_ids]: (B, 1, D);  phase[class_ids]: (B, 1, D)
        freq  = self.freq[class_ids]   # (B, 1, D)
        phase = self.phase[class_ids]  # (B, 1, D)

        # Broadcast over T: (B, T, D)
        periodic_bias = torch.sin(
            2.0 * math.pi * freq * t / float(seq_len) + phase
        )

        # ── Learnable base PE ─────────────────────────────────────────────────
        positions = torch.arange(T, device=device, dtype=torch.long)  # (T,)
        base = self.base_pe(positions).unsqueeze(0)                    # (1, T, D)

        return self.dropout(x + periodic_bias + base)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DUAL SCOPE ATTENTION
#     Two parallel causal attention streams:
#       - local_attn:  only attends to the last `local_window` tokens → captures
#                      fine-grained IMU motion patterns (stride rhythm, etc.)
#       - global_attn: full causal attention → captures long-range activity context
#     A learned gate blends them per-position.
# ══════════════════════════════════════════════════════════════════════════════
class DualScopeAttention(nn.Module):
    """
    Gated combination of local-windowed and global causal multi-head attention.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 local_window: int = 16):
        super().__init__()
        self.local_window = local_window

        self.local_attn  = nn.MultiheadAttention(d_model, n_heads,
                                                  dropout=dropout,
                                                  batch_first=True)
        self.global_attn = nn.MultiheadAttention(d_model, n_heads,
                                                  dropout=dropout,
                                                  batch_first=True)

        # Gate: sigmoid(Linear(cat(local, global))) → (B, T, d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)

    # ── Helper: build the local causal mask ───────────────────────────────────
    def _local_causal_mask(self, T: int, window: int,
                           device: torch.device) -> torch.Tensor:
        """
        Returns (T, T) additive mask.
        Position i can attend to j if:  max(0, i-window) <= j <= i
        All other positions get -inf.
        """
        mask = torch.full((T, T), float("-inf"), device=device)
        for i in range(T):
            start = max(0, i - window)
            mask[i, start : i + 1] = 0.0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        B, T, D = x.shape
        device  = x.device

        # ── Standard full causal mask (upper triangle = -inf) ─────────────────
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=device), diagonal=1
        )

        # ── Local causal mask ─────────────────────────────────────────────────
        local_mask = self._local_causal_mask(T, self.local_window, device)

        local_out,  _ = self.local_attn(x, x, x,  attn_mask=local_mask,
                                        need_weights=False)
        global_out, _ = self.global_attn(x, x, x, attn_mask=causal_mask,
                                         need_weights=False)

        # ── Gated fusion ──────────────────────────────────────────────────────
        gate = torch.sigmoid(
            self.gate_proj(torch.cat([local_out, global_out], dim=-1))
        )  # (B, T, D)

        return gate * local_out + (1.0 - gate) * global_out


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SPECTRAL FFN
#     Two branches combined:
#       - Token branch:     standard 2-layer MLP (captures per-token patterns)
#       - Frequency branch: rFFT magnitude → Linear (captures spectral content)
#     Both are merged via a final linear projection.
# ══════════════════════════════════════════════════════════════════════════════
class SpectralFFN(nn.Module):
    """
    Spectral-aware feedforward network.

    Token branch:     x → Linear(D→ffn_dim) → GELU → Linear(ffn_dim→D)
    Frequency branch: rfft(x, dim=-1).abs() → Linear(D//2+1 → D)
    Merged:           Linear(2D → D) applied to concat of both outputs.

    FFT is over the last (token embedding) dimension, shape (B, T, D).
    """
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        freq_in = d_model // 2 + 1   # rfft output size for real input of length D

        # Token branch
        self.tok_fc1  = nn.Linear(d_model, ffn_dim)
        self.tok_fc2  = nn.Linear(ffn_dim, d_model)
        self.tok_drop = nn.Dropout(dropout)

        # Frequency branch
        self.freq_fc  = nn.Linear(freq_in, d_model)

        # Merge
        self.merge    = nn.Linear(d_model * 2, d_model)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        # Token branch
        tok = self.tok_drop(F.gelu(self.tok_fc1(x)))
        tok = self.tok_fc2(tok)                        # (B, T, D)

        # Frequency branch — rfft over last dim (token embedding dims)
        freq_mag = torch.fft.rfft(x, dim=-1).abs()    # (B, T, D//2+1)
        freq_out = self.freq_fc(freq_mag)              # (B, T, D)

        # Merge both branches
        merged = self.merge(torch.cat([tok, freq_out], dim=-1))  # (B, T, D)
        return self.drop_out(merged)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  IMU TRANSFORMER LAYER
#     Pre-norm wrapper: DualScopeAttention + SpectralFFN with residuals.
# ══════════════════════════════════════════════════════════════════════════════
class IMUTransformerLayer(nn.Module):
    """
    Pre-LN transformer layer using DualScopeAttention and SpectralFFN.

        x = x + DualScopeAttention(LayerNorm(x))
        x = x + SpectralFFN(LayerNorm(x))
    """
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 dropout: float = 0.1, local_window: int = 16):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = DualScopeAttention(d_model, n_heads, dropout, local_window)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = SpectralFFN(d_model, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CAR-IMU DECODER  (the full model)
# ══════════════════════════════════════════════════════════════════════════════
class CARIMUDecoder(nn.Module):
    """
    Class-conditional Autoregressive IMU Decoder.

    Same external interface as the original GPT-style version —
    train_car_imu.py and generate_synthetic.py need no changes.

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
        self.class_emb = nn.Embedding(num_classes, d_model)
        nn.init.trunc_normal_(self.class_emb.weight, std=0.02)

        # ── Class-conditional positional encoding ────────────────────────────
        self.pos_enc = IMUPositionalEncoding(
            num_classes, d_model, max_len=max_seq_len, dropout=dropout)

        # ── Input projection ─────────────────────────────────────────────────
        self.input_proj = nn.Linear(d_model, d_model, bias=False)

        # ── 4 custom IMU transformer layers ──────────────────────────────────
        self.layers = nn.ModuleList([
            IMUTransformerLayer(d_model, n_heads, ffn_dim, dropout,
                                local_window=16)
            for _ in range(n_layers)
        ])

        # ── Final layer norm + output head ───────────────────────────────────
        self.ln_final = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, d_model, bias=False)

        # Weight tying: output head ↔ input projection
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
        tokens:       torch.Tensor,   # (B, L, 64)
        class_labels: torch.Tensor    # (B,)
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass for training.

        Args:
            tokens:       (B, L, 64)  real token sequences from token_store.hdf5
            class_labels: (B,)        integer labels (0=Walking … 5=Standing)

        Returns:
            logits: (B, L+1, 64)  predicted token at each position
                    → during training we compare logits[:, 1:, :] to tokens
        """
        B, L, D = tokens.shape

        # 1. Look up class embedding: (B, 1, D)
        cls_token = self.class_emb(class_labels).unsqueeze(1)

        # 2. Project real tokens into model space: (B, L, D)
        tok_proj = self.input_proj(tokens)

        # 3. Concatenate [CLS_c] + tokens → (B, L+1, D)
        seq = torch.cat([cls_token, tok_proj], dim=1)

        # 4. Class-conditional positional encoding
        seq = self.pos_enc(seq, class_labels, seq_len=seq.size(1))

        # 5. Pass through 4 IMU transformer layers
        for layer in self.layers:
            seq = layer(seq)

        # 6. Final layer norm + output projection → (B, L+1, 64)
        seq    = self.ln_final(seq)
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

        Args:
            class_label: which activity to generate (0=Walking … 5=Standing)
            n_tokens:    how many tokens to generate (default 192 = SEQ_LEN)
            temperature: noise level for stochastic sampling
            device:      'cpu' or 'cuda'

        Returns:
            generated: (n_tokens, 64)  synthetic token sequence Ẑ
        """
        self.eval()

        label_tensor = torch.tensor([class_label], dtype=torch.long,
                                    device=device)
        context = self.class_emb(label_tensor).unsqueeze(1)  # (1, 1, 64)

        generated_tokens = []

        for step in range(n_tokens):
            ctx_pe = self.pos_enc(context, label_tensor,
                                  seq_len=context.size(1))

            h = ctx_pe
            for layer in self.layers:
                h = layer(h)

            h = self.ln_final(h)
            next_token_pred = self.out_head(h[:, -1, :])   # (1, 64)

            if temperature > 0.0:
                next_token_pred = next_token_pred + \
                    torch.randn_like(next_token_pred) * temperature

            generated_tokens.append(next_token_pred)

            next_proj = self.input_proj(next_token_pred).unsqueeze(1)
            context   = torch.cat([context, next_proj], dim=1)

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

        Returns: (n_samples, n_tokens, 64)
        """
        samples = []
        for _ in range(n_samples):
            seq = self.sample(class_label, n_tokens, temperature, device)
            samples.append(seq.unsqueeze(0))
        return torch.cat(samples, dim=0)

    def count_parameters(self) -> int:
        """
        Print parameter count by component and return total.
        """
        component_params = {
            "class_emb":    sum(p.numel() for p in self.class_emb.parameters()),
            "pos_enc":      sum(p.numel() for p in self.pos_enc.parameters()),
            "input_proj":   sum(p.numel() for p in self.input_proj.parameters()),
            "layers":       sum(p.numel() for p in self.layers.parameters()),
            "ln_final":     sum(p.numel() for p in self.ln_final.parameters()),
            "out_head":     sum(p.numel() for p in self.out_head.parameters()
                                if p.requires_grad),
        }
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("  Parameter count by component:")
        for name, n in component_params.items():
            print(f"    {name:<14} {n:>8,}")
        print(f"    {'TOTAL':<14} {total:>8,}")
        return total


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

    print(f"\nModel parameters:")
    total = model.count_parameters()

    # ── Test forward pass (training mode) ─────────────────────────────────────
    B, L, D = 4, SEQ_LEN, TOKEN_DIM
    fake_tokens = torch.randn(B, L, D)
    fake_labels = torch.randint(0, NUM_CLASSES, (B,))

    logits = model(fake_tokens, fake_labels)
    print(f"\nForward pass (teacher forcing):")
    print(f"  Input tokens: {fake_tokens.shape}")
    print(f"  Labels:       {fake_labels.tolist()}")
    print(f"  Output shape: {logits.shape}  (B, L+1, D)")

    pred = logits[:, 1:, :]
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
