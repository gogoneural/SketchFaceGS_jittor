# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Xiaodong Gu & Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-1 17:49:25
# @Function      : transformer_block

import pdb
import logging
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import jittor as jt
from jittor import nn

logger = logging.getLogger(__name__)


class TransformerDecoder(nn.Module):
    """
    Transformer blocks that process the input and optionally use condition and modulation.
    """

    motion_embed_type = ["sd3_mm_cond", "sd3_mm_bh_cond"]

    def __init__(
        self,
        block_type: str,
        num_layers: int,
        num_heads: int,
        inner_dim: int,
        cond_dim: int = None,
        mod_dim: int = None,
        gradient_checkpointing=False,
        eps: float = 1e-6,
        pos_num = 3660+1024,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.block_type = block_type

        if (
            block_type == "sd3_cond"
            or block_type == "sd3_mm_cond"
            or block_type == "sd3_mm_bh_cond"
        ):
            # dual_attention_layers = list(range(num_layers//2))
            dual_attention_layers = []
            self.layers = nn.ModuleList(
                [
                    self._block_fn(inner_dim, cond_dim, mod_dim,pos_num)(
                        num_heads=num_heads,
                        eps=eps,
                        context_pre_only=i == num_layers - 1,
                        use_dual_attention=(
                            True if i in dual_attention_layers else False
                        ),
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    self._block_fn(inner_dim, cond_dim, mod_dim,pos_num)(
                        num_heads=num_heads,
                        eps=eps,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.norm = nn.LayerNorm(inner_dim, eps=eps)

        if self.block_type in [
            "cogvideo_cond",
            "sd3_cond",
            "sd3_mm_cond",
            "sd3_mm_bh_cond",
        ]:
            self.linear_cond_proj = nn.Linear(cond_dim, inner_dim)

    @property
    def block_type(self):
        return self._block_type

    @block_type.setter
    def block_type(self, block_type):
        assert block_type in [
            "basic",
            "cond",
            "mod",
            "cond_mod",
            "sd3_cond",
            "sd3_mm_cond",
            "sd3_mm_bh_cond",
            "cogvideo_cond",
        ], f"Unsupported block type: {block_type}"
        self._block_type = block_type

    def _block_fn(self, inner_dim, cond_dim, mod_dim,pos_num):
        assert inner_dim is not None, f"inner_dim must always be specified"
        if self.block_type == "basic":
            assert (
                cond_dim is None and mod_dim is None
            ), f"Condition and modulation are not supported for BasicBlock"
            from .block import BasicBlock

            logger.debug(f"Using BasicBlock")
            return partial(BasicBlock, inner_dim=inner_dim)
        elif self.block_type == "cond":
            assert (
                cond_dim is not None
            ), f"Condition dimension must be specified for ConditionBlock"
            assert (
                mod_dim is None
            ), f"Modulation dimension is not supported for ConditionBlock"
            from .block import ConditionBlock

            logger.debug(f"Using ConditionBlock")
            return partial(ConditionBlock, inner_dim=inner_dim, cond_dim=cond_dim)
        elif self.block_type == "mod":
            logger.error(f"modulation without condition is not implemented")
            raise NotImplementedError(
                f"modulation without condition is not implemented"
            )
        elif self.block_type == "cond_mod":
            assert (
                cond_dim is not None and mod_dim is not None
            ), f"Condition and modulation dimensions must be specified for ConditionModulationBlock"
            from .block import ConditionModulationBlock

            logger.debug(f"Using ConditionModulationBlock")
            return partial(
                ConditionModulationBlock,
                inner_dim=inner_dim,
                cond_dim=cond_dim,
                mod_dim=mod_dim,
            )
        elif self.block_type == "cogvideo_cond":
            logger.debug(f"Using CogVideoXBlock")
            from LHM.models.transformer_dit_jittor import CogVideoXBlock

            # assert inner_dim == cond_dim, f"inner_dim:{inner_dim}, cond_dim:{cond_dim}"
            return partial(CogVideoXBlock, dim=inner_dim, attention_bias=True)
        elif self.block_type == "sd3_cond":
            logger.debug(f"Using SD3JointTransformerBlock")
            from LHM.models.transformer_dit_jittor import SD3JointTransformerBlock

            return partial(SD3JointTransformerBlock, dim=inner_dim, qk_norm="rms_norm")
        elif self.block_type == "sd3_mm_cond":
            logger.debug(f"Using SD3MMJointTransformerBlock")
            from LHM.models.transformer_dit_jittor import SD3JointTransformerBlock as SD3MMJointTransformerBlock

            return partial(
                SD3MMJointTransformerBlock, dim=inner_dim, qk_norm="rms_norm"
            )
        elif self.block_type == "sd3_mm_bh_cond":
            # logger.debug(f"Using SD3MMJointTransformerBlock")
            from LHM.models.transformer_dit_jittor import SD3BodyHeadMMJointTransformerBlock2

            return partial(
                SD3BodyHeadMMJointTransformerBlock2, dim=inner_dim, qk_norm="rms_norm",pos_num = pos_num,
            )
        else:
            raise ValueError(
                f"Unsupported block type during runtime: {self.block_type}"
            )

    def assert_runtime_integrity(self, x, cond, mod):
        assert x is not None, f"Input tensor must be specified"
        if self.block_type == "basic":
            assert (
                cond is None and mod is None
            ), f"Condition and modulation are not supported for BasicBlock"
        elif "cond" in self.block_type:
            assert (
                cond is not None and mod is None
            ), f"Condition must be specified and modulation is not supported for ConditionBlock"
        elif self.block_type == "mod":
            raise NotImplementedError(
                f"modulation without condition is not implemented"
            )
        else:
            assert (
                cond is not None and mod is not None
            ), f"Condition and modulation must be specified for ConditionModulationBlock"

    def forward_layer(self, layer, x, cond, mod):
        if self.block_type == "basic":
            return layer(x)
        elif self.block_type == "cond":
            return layer(x, cond)
        elif self.block_type == "mod":
            return layer(x, mod)
        else:
            return layer(x, cond, mod)

    def execute(
        self,
        x,
        cond=None,
        mod=None,
        temb=None,
    ):
        """
        Forward pass of the transformer model.
        Args:
            x: Input tensor of shape [N, L, D].
            cond: Conditional tensor of shape [N, L_cond, D_cond] or None.
            mod: Modulation tensor of shape [N, D_mod] or None.
            temb: Modulation tensor of shape [N, D_mod] or None.
        Returns:
            Output tensor of shape [N, L, D].
        """

        self.assert_runtime_integrity(x, cond, mod)

        if self.block_type in [
            "cogvideo_cond",
            "sd3_cond",
            "sd3_mm_cond",
            "sd3_mm_bh_cond",
        ]:
            cond = self.linear_cond_proj(cond)
            for layer in self.layers:
                x, cond = layer(
                    hidden_states=x,
                    encoder_hidden_states=cond,
                    temb=temb,
                )

            x = self.norm(x)
        else:
            for layer in self.layers:
                x = self.forward_layer(layer, x, cond, mod)
            x = self.norm(x)

        return x
