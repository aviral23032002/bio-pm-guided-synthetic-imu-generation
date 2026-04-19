"""
BioPM (Biological Primitives Model) — cleaned for student use.

This file contains the full BioPM / 50MR model architecture:
  - ConvEncode:          1-D CNN that encodes each movement-element patch
  - RelPosMultiheadAttention: Multi-head attention with relative positional bias
  - RelPosTransformerEncoderLayer: Transformer block using the above attention
  - TimeSeriesTransformer (encoder_acc): stacks ConvEncode + axis/duration
        embeddings + 5 RelPos transformer layers → per-patch tokens (B, L, 64)
  - GravityCNNEncoder:   1-D CNN for the low-pass gravity stream → (B, 64)
  - TransformerClassifier: fusion head (not needed for feature extraction)
  - BioPMModel (= DualStreamTimesSeriesTransformerClassifier): full model

For feature extraction you only need encoder_acc and optionally
encoder_gravity.  The classifier head is included so the checkpoint
loads without errors, but you should not use its output as a feature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Polyfills for older PyTorch versions
# ---------------------------------------------------------------------------
if not hasattr(torch, "nan_to_num"):
    def _nan_to_num(x, nan=0.0, posinf=None, neginf=None):
        if not torch.is_floating_point(x):
            return x
        out = x.clone()
        if nan is not None:
            out = torch.where(torch.isnan(out),
                              torch.as_tensor(nan, dtype=out.dtype, device=out.device), out)
        finfo = torch.finfo(out.dtype)
        pos_val = finfo.max if posinf is None else posinf
        neg_val = finfo.min if neginf is None else neginf
        inf_mask = torch.isinf(out)
        out = torch.where(inf_mask & (out > 0),
                          torch.as_tensor(pos_val, dtype=out.dtype, device=out.device), out)
        out = torch.where(inf_mask & (out < 0),
                          torch.as_tensor(neg_val, dtype=out.dtype, device=out.device), out)
        return out
    torch.nan_to_num = _nan_to_num

if not hasattr(torch, "nanmedian"):
    def _nanmedian(x, dim=None, keepdim=False):
        if dim is None:
            x = x.reshape(-1)
            dim = 0
        mask = ~torch.isnan(x)
        filled = torch.where(mask, x, torch.tensor(float('inf'), device=x.device, dtype=x.dtype))
        sorted_vals, sorted_idx = torch.sort(filled, dim=dim, descending=False)
        counts = mask.sum(dim=dim)
        k = torch.clamp(counts - 1, min=0) // 2
        index_shape = list(sorted_vals.shape)
        index_shape[dim] = 1
        k_exp = k.reshape(index_shape).long()
        vals = torch.gather(sorted_vals, dim, k_exp)
        idxs = torch.gather(sorted_idx, dim, k_exp)
        if not keepdim:
            vals = vals.squeeze(dim)
            idxs = idxs.squeeze(dim)
        zero_mask = counts == 0
        if zero_mask.any():
            if keepdim:
                zero_mask = zero_mask.unsqueeze(dim)
            vals = vals.clone()
            vals[zero_mask] = float('nan')
            idxs = idxs.clone()
            idxs[zero_mask] = 0
        return vals, idxs
    torch.nanmedian = _nanmedian


# ---------------------------------------------------------------------------
# Attention mask helper
# ---------------------------------------------------------------------------
def build_additive_mask(bool_mask: torch.Tensor) -> torch.Tensor:
    m = bool_mask.to(torch.bool)
    return torch.where(m, float('-inf'), 0.0)


# ---------------------------------------------------------------------------
# Relative-position multi-head attention
# ---------------------------------------------------------------------------
class RelPosMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 max_rel_pos: int = 50, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.scaling = self.head_dim ** -0.5
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.max_rel_pos = max_rel_pos
        self.rel_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_rel_pos + 1))
        nn.init.trunc_normal_(self.rel_bias, std=0.1)
        self.use_seq_rel_pos = True
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                patch_indices: Tensor,
                attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                is_causal: bool = False) -> Tensor:
        B, L, D = query.shape
        qkv = self.in_proj(query)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q = q * self.scaling
        attn_scores = torch.einsum("bhld,bhmd->bhlm", q, k)

        if self.use_seq_rel_pos:
            idxs = torch.arange(L, device=query.device)
            rel_pos = idxs[None, :] - idxs[:, None]
            rel_pos_clipped = rel_pos.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
            attn_scores = attn_scores + self.rel_bias[:, rel_pos_clipped]

        deltas = patch_indices[:, 1:] - patch_indices[:, :-1]
        dt, _ = torch.nanmedian(deltas, dim=1)
        dt = torch.where(torch.isfinite(dt) & (dt > 0), dt, torch.tensor(1.0, device=dt.device))
        dt = dt.view(-1, 1, 1)
        rel_time = patch_indices.unsqueeze(2) - patch_indices.unsqueeze(1)
        invalid_mask = torch.isnan(rel_time) | (rel_time < -self.max_rel_pos) | (rel_time > self.max_rel_pos)
        rel_time = torch.where(invalid_mask, torch.tensor(0.0, device=rel_time.device), rel_time)
        rel_steps = (rel_time / dt).round()
        M = self.max_rel_pos
        rel_idx = rel_steps.clamp(-M, M).long() + M
        B_, H, N, _ = attn_scores.shape
        expanded_idx = rel_idx.unsqueeze(1).expand(B_, H, N, N)
        bias_table = self.rel_bias.unsqueeze(0).unsqueeze(2).expand(B_, H, N, 2 * M + 1)
        bias = torch.gather(bias_table, dim=3, index=expanded_idx)
        attn_scores = attn_scores + bias

        if is_causal:
            causal_mask = torch.triu(torch.ones(L, L, device=query.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(causal_mask[None, None], float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                additive = build_additive_mask(attn_mask)
            else:
                additive = attn_mask
            attn_scores = attn_scores + additive.unsqueeze(0).unsqueeze(0)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2).bool()
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.einsum("bhlm,bhmd->bhld", attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer encoder layer with relative position bias
# ---------------------------------------------------------------------------
class RelPosTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, max_rel_pos: int = 50,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        self.self_attn = RelPosMultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            max_rel_pos=max_rel_pos, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, src: Tensor, patch_indices: Tensor,
                src_mask: Optional[Tensor] = None, is_causal: bool = False,
                src_key_padding_mask: Optional[Tensor] = None):
        x = src
        sa = self.self_attn(x, x, x, patch_indices, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        x = x + self.drop1(sa)
        x = self.norm1(x)
        ff = self.linear2(self.drop2(self.activation(self.linear1(x))))
        x = x + ff
        return self.norm2(x)


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------
class AvgMaxPool1d(nn.Module):
    def __init__(self, K: int):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(K)
        self.max = nn.AdaptiveMaxPool1d(K)

    def forward(self, x):
        return torch.cat([self.avg(x), self.max(x)], dim=1)


# ---------------------------------------------------------------------------
# ConvEncode: 1-D CNN per movement-element patch  (B*L, 1, 32) → (B*L, 60)
# ---------------------------------------------------------------------------
class ConvEncode(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 60, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(60), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        # x: (B, num_patches, patch_dim=32)
        batch_size, num_patches, patch_dim = x.shape
        x_reshaped = x.reshape(-1, 1, patch_dim)
        x_reshaped = self.conv(x_reshaped)
        return x_reshaped.reshape(batch_size, num_patches, -1)


# ---------------------------------------------------------------------------
# GravityCNNEncoder: (B, 400, 3) → (B, 64)
# ---------------------------------------------------------------------------
DROPOUT_P = 0.02

class GravityCNNEncoder(nn.Module):
    """1-D residual-style CNN for 3-axis gravity windows."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 64,
                 dropout_p: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=1, padding=2,
                      bias=False, padding_mode='circular'),
            nn.GroupNorm(4, 16), nn.GELU(), nn.Dropout2d(p=DROPOUT_P),
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1,
                      bias=False, padding_mode='circular'),
            nn.GroupNorm(8, 32), nn.GELU(), nn.Dropout2d(p=DROPOUT_P),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1,
                      bias=False, padding_mode='circular'),
            nn.GroupNorm(16, 64), nn.GELU(),
        )
        K = 12
        self.pool = AvgMaxPool1d(K)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(128 * K, embed_dim),
            nn.GELU(),
        )
        self.out_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3) → transpose to (B, 3, T)
        x = x.transpose(1, 2)
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# TimeSeriesTransformer (encoder_acc)
# Input:  movement-element patches + position + axis/duration metadata
# Output: (B, L, 64) token representations
# ---------------------------------------------------------------------------
class TimeSeriesTransformer(nn.Module):
    """
    Accelerometer movement-element encoder.

    Takes as input:
      - patches:              (B, L, 32) normalized movement elements
      - patch_indices:        (B, L) fractional position within window [0,1]
      - mask_info:            (B, L) float mask (1 = masked, 0 = visible)
      - additional_embedding: (B, L, >=2) where [:,:,0]=axis, [:,:,1]=duration

    Returns: (B, L, 64) contextual token embeddings.
    """

    def __init__(self):
        super().__init__()
        D = 64
        self.mask_token = nn.Parameter(torch.randn(D))
        nn.init.trunc_normal_(self.mask_token, std=0.2)
        self.conv_encode = ConvEncode(1)
        self.axis_embedding = nn.Embedding(num_embeddings=4, embedding_dim=3)

        self.transformer_encoder_points_within_segment = nn.ModuleList([
            RelPosTransformerEncoderLayer(
                d_model=64, nhead=4, max_rel_pos=15, dropout=DROPOUT_P,
            ) for _ in range(5)
        ])

        self.pos_emb_net = nn.Sequential(
            nn.Linear(1, D), nn.GELU(), nn.Linear(D, D))
        self.post_pos_ln = nn.LayerNorm(D)

    def forward(self, patches, patch_indices, mask_info, additional_embedding,
                is_causal=False):
        padding_mask = ~torch.isnan(patches).any(dim=-1)
        patches = torch.nan_to_num(patches, nan=25)
        additional_embedding = torch.nan_to_num(additional_embedding, nan=3)[:, :, :2]

        patches = self.conv_encode(patches)  # (B, L, 60)

        axis_emb = self.axis_embedding(additional_embedding[:, :, 0].long())  # (B, L, 3)
        patches = torch.cat([patches, axis_emb], dim=-1)  # (B, L, 63)
        duration = additional_embedding[:, :, 1].unsqueeze(-1)  # (B, L, 1)
        patches = torch.cat([patches, duration], dim=-1)  # (B, L, 64)

        mask_info_expanded = mask_info.unsqueeze(-1).expand_as(patches).bool()
        patches = torch.where(mask_info_expanded, self.mask_token, patches)

        patch_indices = torch.nan_to_num(patch_indices, nan=0)
        pos_embedding = self.pos_emb_net(patch_indices.unsqueeze(-1))
        patches = patches + pos_embedding
        patches = self.post_pos_ln(patches)

        combined_mask = ~padding_mask
        x = patches
        for layer in self.transformer_encoder_points_within_segment:
            x = layer(x, patch_indices, is_causal=is_causal,
                      src_key_padding_mask=combined_mask)
        return x


# ---------------------------------------------------------------------------
# Pooling used during feature extraction
# ---------------------------------------------------------------------------
def masked_mean(x, mask=None):
    """Mean-pool over the sequence dimension.  x: (B, L, D) → (B, D)."""
    return x.mean(dim=1)


def masked_mean_std(x, mask=None):
    """Mean + std pool over sequence.  x: (B, L, D) → (B, 2*D)."""
    return torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1)


# ---------------------------------------------------------------------------
# Classifier head (needed for checkpoint compat, not for feature extraction)
# ---------------------------------------------------------------------------
class AttnPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x, mask):
        mask_bool = mask.bool() if mask.dtype == torch.bool else (mask > 0.5)
        logits = self.score(x).squeeze(-1)
        logits = logits - logits.max(dim=1, keepdim=True).values
        exp_logits = torch.exp(logits) * mask_bool.float()
        weights = exp_logits / exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class TransformerClassifier(nn.Module):
    def __init__(self, N, D, n_classes):
        super().__init__()
        self.hidden_dim = 1
        self.classifier2 = nn.Sequential(
            nn.Linear(D * 1, D // 2),
            nn.ReLU(inplace=True),
            nn.Linear(D // 2, D // 4),
            nn.ReLU(inplace=True),
            nn.Linear(D // 4, n_classes),
        )
        self.dropout1 = nn.Dropout(DROPOUT_P)
        self.dropout2 = nn.Dropout(DROPOUT_P)
        self.attn_pool = AttnPool(D // 2, hidden=128)

    def forward(self, x_acc, x_gravity, mask):
        x_acc = masked_mean(x_acc, mask)
        x_gravity = self.dropout2(x_gravity)
        x = torch.cat((x_acc, x_gravity), dim=-1)
        return self.classifier2(x)


# ---------------------------------------------------------------------------
# BioPMModel  (= DualStreamTimesSeriesTransformerClassifier)
# ---------------------------------------------------------------------------
class BioPMModel(nn.Module):
    """
    Full BioPM dual-stream model.

    Streams:
      1. encoder_acc   (TimeSeriesTransformer): movement-element patches → tokens
      2. encoder_gravity (GravityCNNEncoder):   raw gravity window → 64-d vector
      3. classifier    (TransformerClassifier):  fuses both → class logits

    For feature extraction, use extract_features() from the inference module
    instead of calling forward().
    """

    def __init__(self, n_classes: int = 11):
        super().__init__()
        self.encoder_acc = TimeSeriesTransformer()
        self.encoder_gravity = GravityCNNEncoder(dropout_p=0.025)
        self.classifier = TransformerClassifier(192, 128, n_classes)

    def forward(self, batch_me_acc, batch_me_gravity,
                pos_info_acc, additional_embedding_acc, mask):
        acc_feat = self.encoder_acc(batch_me_acc, pos_info_acc, mask,
                                    additional_embedding_acc)
        gravity_feat = self.encoder_gravity(batch_me_gravity)
        padding_mask_acc = 1.0 * (~torch.isnan(batch_me_acc).any(dim=-1))
        logits = self.classifier(acc_feat, gravity_feat, padding_mask_acc)
        return logits

    def load_encoder_weights(self, checkpoint_path: str, strict: bool = False):
        """Load pretrained 50MR weights into encoder_acc only."""
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE,
                                weights_only=False)
        keys = self.encoder_acc.load_state_dict(checkpoint, strict=strict)
        print(f"Loaded encoder_acc weights from {checkpoint_path}")
        if keys.missing_keys:
            print(f"  missing keys: {keys.missing_keys}")
        if keys.unexpected_keys:
            print(f"  unexpected keys: {keys.unexpected_keys}")
        return keys


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------
def load_pretrained_encoder(checkpoint_path: str, n_classes: int = 11,
                            device: str = "cpu") -> BioPMModel:
    """
    Instantiate BioPMModel and load pretrained encoder_acc weights.

    Args:
        checkpoint_path: path to submovement_transformer_50MR/checkpoint.pt
        n_classes: number of downstream classes (only affects classifier head)
        device: 'cpu' or 'cuda'

    Returns:
        model in eval mode on the specified device
    """
    model = BioPMModel(n_classes=n_classes)
    model.to(device, dtype=torch.float)
    model.load_encoder_weights(checkpoint_path)
    model.eval()
    return model
