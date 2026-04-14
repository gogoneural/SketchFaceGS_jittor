# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom replacement for `torch.nn.functional.conv2d` that supports
arbitrarily high order gradients with zero performance penalty."""

import jittor as jt

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # if _should_use_custom_op(input):
    #     return _conv2d_gradfix(transpose=False, weight_shape=weight.shape, stride=stride, padding=padding, output_padding=0, dilation=dilation, groups=groups).apply(input, weight, bias)
    return jt.nn.conv2d(input, weight=weight, bias=bias, stride=stride, padding=padding, groups=groups, dilation=dilation)

def _normalize_param(p):
    if isinstance(p, (list, tuple)):
        return tuple(p)
    return p

def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    return jt.nn.conv_transpose2d(input, weight=weight, bias=bias, stride=_normalize_param(stride), padding=_normalize_param(padding), output_padding=_normalize_param(output_padding), groups=groups, dilation=_normalize_param(dilation))
