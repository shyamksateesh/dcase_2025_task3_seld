"""
model.py  (Transformer variant)

Drop-in replacement for the Yeow et al. biGRU + MHSA backbone.
Only the sequence-modeling block changes:
    BEFORE: GRU(bidirectional) → element-wise multiply halves → 2x MHSA layers
    AFTER:  Transformer Encoder (N layers, RoPE positional encoding)

Everything else — ConvBlock, FNN decoder, output shape — is identical to the
original model.py so loss.py / data_generator.py / main.py need no changes.

Architecture choices (from project proposal):
    d_model       = params['rnn_size']  (128 in baseline, set to 256 for Yeow-scale)
    nhead         = 4  (ablate: 8)
    num_layers    = 2  (ablate: 3)
    dim_feedforward = 512  (ablate: 256, 1024)
    dropout       = params['dropout']  (ablate: 0.1, 0.2, 0.3)
    positional_encoding = RoPE  (ablate: sinusoidal)

Controlled via params keys:
    params['use_transformer'] = True      <- switch between GRU and Transformer
    params['tr_nhead']        = 4
    params['tr_layers']       = 2
    params['tr_ff_dim']       = 512
    params['tr_rope']         = True      <- False = sinusoidal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ─────────────────────────────────────────────────────────────────────────────
# Positional encodings
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    Encodes *relative* position by rotating Q and K vectors before the dot-product.
    Reference: Su et al. 2021 "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    def __init__(self, dim):
        super().__init__()
        # Inverse frequencies; shape (dim/2,)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def get_cos_sin(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)   # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)              # (T, dim)
        return emb.cos(), emb.sin()

    @staticmethod
    def rotate_half(x):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def apply(self, q, k, cos, sin):
        """
        q, k: (B, H, T, head_dim)
        cos, sin: (T, head_dim) → broadcast over B and H
        """
        cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, T, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin
        return q, k


# ─────────────────────────────────────────────────────────────────────────────
# Transformer encoder with RoPE
# ─────────────────────────────────────────────────────────────────────────────

class RoPEAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.rope = RotaryEmbedding(self.head_dim)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, C = x.shape
        H, hd = self.nhead, self.head_dim

        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)   # (B, H, T, hd)
        k = self.k_proj(x).view(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hd).transpose(1, 2)

        cos, sin = self.rope.get_cos_sin(T, x.device)           # (T, hd)
        q, k = self.rope.apply(q, k, cos, sin)

        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)   # (B, H, T, T)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)          # (B, T, C)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer (more stable than post-norm for audio)."""
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, use_rope=True):
        super().__init__()
        if use_rope:
            self.attn = RoPEAttention(d_model, nhead, dropout)
        else:
            self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.use_rope = use_rope

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-norm residual
        if self.use_rope:
            x = x + self.attn(self.norm1(x))
        else:
            attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
            x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, use_rope=True):
        super().__init__()
        if not use_rope:
            self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        else:
            self.pos_enc = None
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, use_rope)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# Shared ConvBlock (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, pool_size=(5, 4), dropout=0.05):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.pool(x)
        x = self.dropout(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Main SELD model — GRU or Transformer selected via params['use_transformer']
# ─────────────────────────────────────────────────────────────────────────────

class SELDModel(nn.Module):
    """
    SELD model.  When params['use_transformer'] = True, replaces the biGRU + MHSA
    sequence model with a Transformer Encoder (RoPE or sinusoidal positional encoding).
    Everything else is identical to the original Yeow et al. model.
    """
    def __init__(self, params, in_feat_shape):
        super().__init__()
        self.params = params
        self.in_feat_shape = in_feat_shape
        params['nb_mels'] = in_feat_shape[-1]

        nb_conv_filters = [64] + [params['nb_conv_filters']] * int(params['nb_conv_blocks'] - 1)

        # Conv layers (identical to original)
        self.conv_blocks = nn.ModuleList()
        for conv_cnt in range(params['nb_conv_blocks']):
            self.conv_blocks.append(ConvBlock(
                in_channels=nb_conv_filters[conv_cnt - 1] if conv_cnt else self.in_feat_shape[1],
                out_channels=nb_conv_filters[conv_cnt],
                pool_size=(params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]),
                dropout=params['dropout']
            ))

        # Sequence model — GRU or Transformer
        self.gru_input_dim = params['nb_conv_filters'] * int(
            np.floor(params['nb_mels'] / np.prod(params['f_pool_size']))
        )
        self.use_transformer = params.get('use_transformer', False)

        if self.use_transformer:
            d_model = params['rnn_size']
            # Project conv output to d_model if dimensions differ
            if self.gru_input_dim != d_model:
                self.input_proj = nn.Linear(self.gru_input_dim, d_model)
            else:
                self.input_proj = None

            self.transformer = TransformerEncoder(
                d_model=d_model,
                nhead=params.get('tr_nhead', 4),
                num_layers=params.get('tr_layers', 2),
                dim_feedforward=params.get('tr_ff_dim', 512),
                dropout=params.get('dropout', 0.1),
                use_rope=params.get('tr_rope', True),
            )
            seq_out_dim = d_model

        else:
            # Original GRU path
            self.gru = torch.nn.GRU(
                input_size=self.gru_input_dim,
                hidden_size=params['rnn_size'],
                num_layers=params['nb_rnn_layers'],
                batch_first=True,
                dropout=params['dropout'],
                bidirectional=True
            )
            # MHSA layers (original)
            self.mhsa_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=params['rnn_size'],
                    num_heads=params['nb_attn_heads'],
                    dropout=params['dropout'],
                    batch_first=True
                )
                for _ in range(params['nb_self_attn_layers'])
            ])
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(params['rnn_size']) for _ in range(params['nb_self_attn_layers'])
            ])
            seq_out_dim = params['rnn_size']

        # FNN decoder (identical to original)
        self.fnn_list = torch.nn.ModuleList()
        for fc_cnt in range(params['nb_fnn_layers']):
            self.fnn_list.append(nn.Linear(
                params['fnn_size'] if fc_cnt else seq_out_dim,
                params['fnn_size'], bias=True
            ))
            self.fnn_list.append(nn.Dropout(p=params['dropout']))

        if params['multiACCDOA']:
            self.output_dim = params['max_polyphony'] * 3 * params['nb_classes']
        else:
            self.output_dim = 3 * params['nb_classes']

        self.fnn_list.append(nn.Linear(
            params['fnn_size'] if params['nb_fnn_layers'] else seq_out_dim,
            self.output_dim, bias=True
        ))

    def forward(self, audio_feat):
        # Conv backbone  —  (B, C, T, F) → (B, nb_conv_filters, T', F')
        for conv_block in self.conv_blocks:
            audio_feat = conv_block(audio_feat)

        # Reshape to sequence  —  (B, T', nb_conv_filters * F')
        audio_feat = audio_feat.transpose(1, 2).contiguous()
        audio_feat = audio_feat.view(audio_feat.shape[0], audio_feat.shape[1], -1).contiguous()

        if self.use_transformer:
            # Optional projection if conv output dim ≠ d_model
            if self.input_proj is not None:
                audio_feat = self.input_proj(audio_feat)
            audio_feat = self.transformer(audio_feat)            # (B, T', d_model)

        else:
            # Original: biGRU → element-wise multiply of forward/backward halves
            (audio_feat, _) = self.gru(audio_feat)
            audio_feat = (audio_feat[:, :, audio_feat.shape[-1] // 2:] *
                          audio_feat[:, :, :audio_feat.shape[-1] // 2])

            # Original: MHSA layers with residual
            for mhsa, ln in zip(self.mhsa_layers, self.layer_norms):
                audio_feat_in = audio_feat
                audio_feat, _ = mhsa(audio_feat_in, audio_feat_in, audio_feat_in)
                audio_feat = audio_feat + audio_feat_in
                audio_feat = ln(audio_feat)

        # FNN decoder
        for fnn_cnt in range(len(self.fnn_list) - 1):
            audio_feat = self.fnn_list[fnn_cnt](audio_feat)
        audio_feat = F.relu(audio_feat, inplace=True)
        audio_feat = self.fnn_list[-1](audio_feat)

        return audio_feat
