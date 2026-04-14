# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

# ******************************************************************************
#   Code modified by Zexin He in 2023-2024.
#   Modifications are marked with clearly visible comments
#   licensed under the Apache License, Version 2.0.
# ******************************************************************************

import logging
from typing import Callable, List, Any, Tuple, Dict

import jittor as jt
from jittor import nn

from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


logger = logging.getLogger("dinov2")


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def execute(self, x: jt.Var) -> jt.Var:
        def attn_residual_func(x: jt.Var) -> jt.Var:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: jt.Var) -> jt.Var:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.is_training() and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.is_training() and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


# ********** Modified by Zexin He in 2023-2024 **********
# Override execute with modulation input
class BlockWithModulation(Block):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def execute(self, x: jt.Var, mod: jt.Var) -> jt.Var:
        def attn_residual_func(x: jt.Var, mod: jt.Var) -> jt.Var:
            return self.ls1(self.attn(self.norm1(x, mod)))

        def ffn_residual_func(x: jt.Var, mod: jt.Var) -> jt.Var:
            return self.ls2(self.mlp(self.norm2(x, mod)))

        if self.is_training() and self.sample_drop_ratio > 0.1:
            raise NotImplementedError("Modulation with drop path ratio larger than 0.1 is not supported yet")
        elif self.is_training() and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, mod))
            x = x + self.drop_path1(ffn_residual_func(x, mod))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, mod)
            x = x + ffn_residual_func(x, mod)
        return x
# ********************************************************


def drop_add_residual_stochastic_depth(
    x: jt.Var,
    residual_func: Callable[[jt.Var], jt.Var],
    sample_drop_ratio: float = 0.0,
) -> jt.Var:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = jt.randperm(b)[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = x_flat.clone()
    x_plus_residual[brange] = x_plus_residual[brange] + residual.cast(x.dtype) * residual_scale_factor
    return x_plus_residual.reshape(x.shape)
