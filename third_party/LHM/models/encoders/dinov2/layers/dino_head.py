# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import jittor as jt
from jittor import nn, init
import math


def _trunc_normal_(tensor, std=0.02):
    """Truncated normal initialization (jittor version)."""
    with jt.no_grad():
        size = tensor.shape
        tmp = jt.empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1)[1]
        tensor_data = tmp.gather(-1, ind.unsqueeze(-1)).squeeze(-1)
        tensor_data = tensor_data * std
        tensor.update(tensor_data)
    return tensor


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self._init_weights_all()
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def _init_weights_all(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def execute(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == 'float16' else 1e-12
        x = x / (x.norm(dim=-1, keepdim=True, p=2).clamp(min=eps))
        x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)
