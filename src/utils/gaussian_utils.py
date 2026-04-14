import copy
from typing import Optional

import jittor as jt

from scene.gaussian_model import GaussianModel
from gghead_jittor.util.activation import mip_tanh, mip_sigmoid
from utils.sh_utils import C0

def _apply_color_activation(value):

    color_value = value[..., :3]  # First 3 channels are always color values
    color_value = mip_tanh(color_value, overshoot=0.001)
    color_value = color_value * (0.5 / C0)  # Force colors between [-1.78, 1.78]

    # TODO: SH bands have the same scaling as color bands
    sh_value = value[..., 3:]
    sh_value = mip_tanh(sh_value, overshoot=0.001)
    sh_value = sh_value * (0.5 / C0)  # Force colors between [-1.78, 1.78]

    value = jt.concat([color_value, sh_value], dim=-1)

    return value

def _apply_opacity_activation(value):
    return mip_sigmoid(value, overshoot=0.001, clamp=False)

def clone_gaussian_model(source):
    """Clone a GaussianModel-like object.

    This does a best-effort shallow/deep copy of attributes. All torch.Tensor
    attributes are cloned and moved to `device`/`dtype` when specified. Other
    attributes are deepcopy'd when safe. The returned object will be an
    instance of the same class when possible.

    Args:
        src_model: instance of `GaussianModel` (or similar object).
        device: optional `torch.device` (or string) to move tensors to.
        dtype: optional `torch.dtype` to cast tensors to.

    Returns:
        A new object containing copied attributes. If the class constructor
        raises or is unavailable, returns a deepcopy fallback where tensors
        have been cloned/moved.
    """
   
    # Initialize new model
    target = GaussianModel(sh_degree = 1)
    
    # Clone geometry and attributes – stop_grad() breaks the computation graph
    # so that subsequent jt.gc() cannot free the source tensors' backing memory.
    target._xyz = source._xyz.clone().contiguous().stop_grad()
    target._features_dc = source._features_dc.clone().contiguous().stop_grad()
    target._features_rest = source._features_rest.clone().contiguous().stop_grad()
    target._scaling = source._scaling.clone().contiguous().stop_grad()
    target._rotation = source._rotation.clone().contiguous().stop_grad()
    target._opacity = source._opacity.clone().contiguous().stop_grad()
    
    # Copy setup attributes
    target.active_sh_degree = 1
    target.opacity_activation = _apply_opacity_activation

    # Required by renderer
    target.screenspace_points = jt.zeros_like(target._xyz) + 0
    target.color_activation = _apply_color_activation

    # Materialize all lazy ops before returning
    jt.sync_all()

    return target




__all__ = ["clone_gaussian_model"]
