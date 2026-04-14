import jittor as jt
from jittor import nn
import math

from LHM.models.rendering.utils.typing import *


def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        def _lin2srgb(x):
            clamped = jt.clamp(x, min_v=0.0031308)
            return jt.clamp(jt.ternary(x > 0.0031308,
                jt.pow(clamped, 1.0 / 2.4) * 1.055 - 0.055,
                12.92 * x), min_v=0.0, max_v=1.0)
        return _lin2srgb
    elif name == "exp":
        return lambda x: jt.exp(x)
    elif name == "shifted_exp":
        return lambda x: jt.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: jt.sigmoid(x)
    elif name == "tanh":
        return lambda x: jt.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: nn.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    else:
        try:
            return getattr(nn, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_neurons: int,
        n_hidden_layers: int,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__()
        layers = [
            self.make_linear(
                dim_in, n_neurons, is_first=True, is_last=False, bias=bias
            ),
            self.make_activation(activation),
        ]
        for i in range(n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    n_neurons, n_neurons, is_first=False, is_last=False, bias=bias
                ),
                self.make_activation(activation),
            ]
        layers += [
            self.make_linear(
                n_neurons, dim_out, is_first=False, is_last=True, bias=bias
            )
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(output_activation)

    def execute(self, x):
        x = self.layers(x)
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last, bias=True):
        layer = nn.Linear(dim_in, dim_out, bias=bias)
        return layer

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "silu":
            return nn.SiLU()
        else:
            raise NotImplementedError


def trunc_exp(x):
    # Jittor equivalent of truncated exponential
    # For inference, forward pass is just exp; for backward, gradient is clamped
    return jt.exp(jt.clamp(x, max_v=15.0))
