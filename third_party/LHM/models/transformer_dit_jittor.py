# -*- coding: utf-8 -*-
# Jittor reimplementation of transformer_dit.py
# Replaces diffusers dependencies with pure Jittor implementations

import math
import pdb
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import jittor as jt
from jittor import nn
import numpy as np


# ============================================================
# Pure Jittor reimplementations of diffusers components
# ============================================================

class GEGLU(nn.Module):
    """GELU activation with gating, used in FeedForward."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def execute(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * nn.gelu(gate)


class ApproximateGELU(nn.Module):
    """Approximate GELU activation."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def execute(self, x):
        x = self.proj(x)
        return x * jt.sigmoid(1.702 * x)


class FeedForward(nn.Module):
    """Feed-forward network used in transformer blocks."""
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0,
                 activation_fn="geglu", final_dropout=False,
                 inner_dim=None, bias=True):
        super().__init__()
        inner_dim = inner_dim or int(dim * mult)
        dim_out = dim_out or dim

        if activation_fn == "gelu":
            act_fn = nn.Sequential(nn.Linear(dim, inner_dim, bias=bias), nn.GELU())
        elif activation_fn == "gelu-approximate":
            act_fn = nn.Sequential(
                nn.Linear(dim, inner_dim, bias=bias),
                nn.GELU()  # Jittor GELU is approximate by default
            )
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")

        self.net = nn.ModuleList([])
        # activation
        self.net.append(act_fn)
        # dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def execute(self, x):
        for module in self.net:
            x = module(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = jt.ones(dim)

    def execute(self, x):
        norm = jt.sqrt(jt.mean(x * x, dim=-1, keepdims=True) + self.eps)
        return x / norm * self.weight


class JointAttention(nn.Module):
    """
    Joint attention module that processes both hidden_states and encoder_hidden_states.
    Reimplements diffusers Attention + JointAttnProcessor2_0.
    """
    def __init__(self, query_dim, num_heads, dim_head, out_dim=None,
                 context_pre_only=False, bias=True, qk_norm=None, eps=1e-6,
                 added_kv_proj_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head
        self.context_pre_only = context_pre_only
        out_dim = out_dim or query_dim

        # Query, Key, Value projections for hidden_states
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)

        # Additional projections for encoder_hidden_states (context)
        kv_dim = added_kv_proj_dim if added_kv_proj_dim is not None else query_dim
        self.add_q_proj = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.add_k_proj = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.add_v_proj = nn.Linear(kv_dim, self.inner_dim, bias=bias)

        # Output projections
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, out_dim, bias=bias),
            nn.Dropout(0.0),
        )
        if not context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, out_dim, bias=bias)

        # QK normalization
        self.norm_q = None
        self.norm_k = None
        self.norm_added_q = None
        self.norm_added_k = None
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
            if added_kv_proj_dim is not None:
                self.norm_added_q = RMSNorm(dim_head, eps=eps)
                self.norm_added_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps)
            if added_kv_proj_dim is not None:
                self.norm_added_q = nn.LayerNorm(dim_head, eps=eps)
                self.norm_added_k = nn.LayerNorm(dim_head, eps=eps)

        self.scale = dim_head ** -0.5

    def execute(self, hidden_states, encoder_hidden_states=None, **kwargs):
        batch_size = hidden_states.shape[0]
        residual_size = hidden_states.shape[1]

        # Project hidden_states
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # QK normalization for hidden_states
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # Project encoder_hidden_states
        encoder_query = self.add_q_proj(encoder_hidden_states)
        encoder_key = self.add_k_proj(encoder_hidden_states)
        encoder_value = self.add_v_proj(encoder_hidden_states)

        # Reshape encoder projections
        encoder_query = encoder_query.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        encoder_key = encoder_key.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        encoder_value = encoder_value.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # QK normalization for encoder_hidden_states (norm_added_q/k)
        if self.norm_added_q is not None:
            encoder_query = self.norm_added_q(encoder_query)
        if self.norm_added_k is not None:
            encoder_key = self.norm_added_k(encoder_key)

        # Concatenate after normalization
        query = jt.concat([query, encoder_query], dim=2)
        key = jt.concat([key, encoder_key], dim=2)
        value = jt.concat([value, encoder_value], dim=2)

        # Scaled dot-product attention
        attn_weight = jt.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weight = nn.softmax(attn_weight, dim=-1)
        hidden_out = jt.matmul(attn_weight, value)

        # Reshape back
        hidden_out = hidden_out.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)

        # Split outputs
        hidden_states_out = hidden_out[:, :residual_size]
        encoder_hidden_states_out = hidden_out[:, residual_size:]

        # Output projections
        hidden_states_out = self.to_out[0](hidden_states_out)
        hidden_states_out = self.to_out[1](hidden_states_out)

        if not self.context_pre_only:
            encoder_hidden_states_out = self.to_add_out(encoder_hidden_states_out)

        return hidden_states_out, encoder_hidden_states_out


class CogVideoXAttention(nn.Module):
    """
    Attention for CogVideoX block that concatenates hidden and encoder states.
    """
    def __init__(self, query_dim, num_heads, dim_head, bias=False,
                 qk_norm=False, eps=1e-6, out_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim, bias=out_bias),
            nn.Dropout(0.0),
        )

        self.norm_q = None
        self.norm_k = None
        if qk_norm:
            self.norm_q = nn.LayerNorm(dim_head, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps)

    def execute(self, hidden_states, encoder_hidden_states=None, **kwargs):
        batch_size = hidden_states.shape[0]
        text_len = encoder_hidden_states.shape[1]

        # Concatenate for joint processing
        x = jt.concat([encoder_hidden_states, hidden_states], dim=1)

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        query = query.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        attn_weight = jt.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weight = nn.softmax(attn_weight, dim=-1)
        out = jt.matmul(attn_weight, value)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)
        out = self.to_out[0](out)
        out = self.to_out[1](out)

        encoder_out = out[:, :text_len]
        hidden_out = out[:, text_len:]

        return hidden_out, encoder_out


def get_1d_rotary_pos_embed(dim, pos, use_real=True):
    """Generate 1D rotary position embeddings."""
    if isinstance(pos, int):
        pos = jt.arange(pos).float32()
    elif isinstance(pos, np.ndarray):
        pos = jt.array(pos).float32()

    freqs = 1.0 / (10000.0 ** (jt.arange(0, dim, 2).float32() / dim))
    t = pos.unsqueeze(1) * freqs.unsqueeze(0)  # [seq_len, dim//2]

    if use_real:
        cos_emb = jt.cos(t)
        sin_emb = jt.sin(t)
        return (cos_emb, sin_emb)
    else:
        emb = jt.concat([jt.cos(t), jt.sin(t)], dim=-1)
        return emb


# ============================================================
# Transformer Blocks
# ============================================================

def _chunked_feed_forward(ff, hidden_states, chunk_dim, chunk_size):
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}."
        )
    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = jt.concat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


class CogVideoXBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        norm_eps = eps
        num_attention_heads = num_heads
        attention_head_dim = dim // num_attention_heads
        assert attention_head_dim * num_attention_heads == dim

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = CogVideoXAttention(
            query_dim=dim,
            num_heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm=qk_norm,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def execute(
        self,
        hidden_states,
        encoder_hidden_states,
        temb=None,
        image_rotary_emb=None,
    ):
        text_seq_length = encoder_hidden_states.shape[1]

        norm_hidden_states = self.norm1(hidden_states)
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)

        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        hidden_states = hidden_states + attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + attn_encoder_hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = jt.concat(
            [norm_encoder_hidden_states, norm_hidden_states], dim=1
        )
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class SD3JointTransformerBlock(nn.Module):
    """
    Joint Transformer block following the MMDiT architecture (SD3).
    Reimplemented with pure Jittor.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
        pos_num=3660 + 1024,
    ):
        super().__init__()
        num_attention_heads = num_heads
        attention_head_dim = dim // num_attention_heads
        assert attention_head_dim * num_attention_heads == dim

        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only

        self.norm1 = nn.LayerNorm(dim)
        self.norm1_context = nn.LayerNorm(dim)

        self.attn = JointAttention(
            query_dim=dim,
            num_heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            qk_norm=qk_norm,
            eps=eps,
            added_kv_proj_dim=dim,
        )

        if use_dual_attention:
            self.attn2 = JointAttention(
                query_dim=dim,
                num_heads=num_attention_heads,
                dim_head=attention_head_dim,
                out_dim=dim,
                bias=True,
                qk_norm=qk_norm,
                eps=eps,
                added_kv_proj_dim=dim,
            )
        else:
            self.attn2 = None

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
            self.ff_context = FeedForward(
                dim=dim, dim_out=dim, activation_fn="gelu-approximate"
            )
        else:
            self.norm2_context = None
            self.ff_context = None

        self._chunk_size = None
        self._chunk_dim = 0

        self.image_rotary_emb = get_1d_rotary_pos_embed(
            dim=64,
            pos=pos_num,
            use_real=True,
        )

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def execute(
        self,
        hidden_states,
        encoder_hidden_states,
        temb=None,
    ):
        norm_hidden_states = self.norm1(hidden_states)
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2, _ = self.attn2(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
            )
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + ff_output

        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            if self._chunk_size is not None:
                context_ff_output = _chunked_feed_forward(
                    self.ff_context,
                    norm_encoder_hidden_states,
                    self._chunk_dim,
                    self._chunk_size,
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + context_ff_output

        return hidden_states, encoder_hidden_states


class SD3BodyHeadMMJointTransformerBlock2(nn.Module):
    """
    BodyHead Transformer block following the MMDiT architecture (SD3).
    Uses only the head_dit (SD3JointTransformerBlock).
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
        pos_num=3660 + 1024,
    ):
        super().__init__()

        self.head_dit = SD3JointTransformerBlock(
            dim,
            num_heads,
            eps,
            context_pre_only=context_pre_only,
            qk_norm=qk_norm,
            use_dual_attention=use_dual_attention,
            pos_num=pos_num,
        )

    def execute(
        self,
        hidden_states,
        encoder_hidden_states,
        temb=None,
    ):
        """Default, process all as head"""
        head_hidden_states = hidden_states
        head_encoder_hidden_states = encoder_hidden_states

        head_states, head_encoder_hidden_states = self.head_dit(
            head_hidden_states, head_encoder_hidden_states,
        )
        hidden_states = head_states
        encoder_hidden_states = head_encoder_hidden_states
        return hidden_states, encoder_hidden_states
