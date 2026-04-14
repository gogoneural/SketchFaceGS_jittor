import jittor as jt
from jittor import nn
import numpy as np

#----------------------------------------------------------------------------
# Helpers.
#----------------------------------------------------------------------------

def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, (int, np.integer)) for x in padding)
    padding = [int(x) for x in padding]
    if len(padding) == 2:
        px, py = padding
        padding = [px, px, py, py]
    assert len(padding) == 4
    return padding

def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, jt.Var)
    assert 1 <= f.ndim <= 2
    fw = f.shape[-1]
    fh = f.shape[0]
    assert fw >= 1 and fh >= 1
    return fw, fh

#----------------------------------------------------------------------------
# Filter setup.
#----------------------------------------------------------------------------

def setup_filter(f, device=None, normalize=True, flip_filter=False, gain=1, separable=None):
    r"""Convenience function to setup 2D FIR filter for `upfirdn2d()`.

    Args:
        f:           Tensor, numpy array, or python list of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (scalar, same as `[1]`), or
                     `None` (identity).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signals? Default: True.
        flip_filter: Flip the filter? Default: False.
        gain:        Overall scaling factor. Default: 1.
        separable:   Return a separable filter? Default: select automatically.

    Returns:
        Float32 jittor Var of the shape `[filter_height, filter_width]` (non-separable)
        or `[filter_taps]` (separable).
    """
    if f is None:
        f = 1
    f = np.float32(f)
    if f.ndim == 0:
        f = f[np.newaxis]

    if separable is None:
        separable = (f.ndim == 1 and f.size >= 8)
    if f.ndim == 1 and not separable:
        f = np.outer(f, f)

    assert f.ndim in [1, 2]
    assert f.size >= 1
    if f.ndim == 1:
        assert separable

    if normalize:
        f /= f.sum()
    if gain != 1:
        f *= gain
    if flip_filter:
        f = f[::-1] if f.ndim == 1 else f[::-1, ::-1]

    f = jt.array(f.copy()).float32()
    assert f.ndim in [1, 2]
    return f

#----------------------------------------------------------------------------
# Main functional API.
#----------------------------------------------------------------------------

def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    r"""Pad, upsample, filter, and downsample a batch of 2D images.

    Args:
        x:           Float32 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or `None` (identity).
        up:          Integer upsampling factor (default: 1).
        down:        Integer downsampling factor (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, jt.Var) and x.ndim == 4
    if isinstance(up, int):
        upx, upy = up, up
    else:
        upx, upy = up
    if isinstance(down, int):
        downx, downy = down, down
    else:
        downx, downy = down
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    return _upfirdn2d_ref(x, f, upx=upx, upy=upy, downx=downx, downy=downy,
                          padx0=padx0, padx1=padx1, pady0=pady0, pady1=pady1,
                          flip_filter=flip_filter, gain=gain)

#----------------------------------------------------------------------------
# Reference implementation.
#----------------------------------------------------------------------------

def _upfirdn2d_ref(x, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1,
                   flip_filter=False, gain=1):
    """Reference implementation of `upfirdn2d()` using standard Jittor ops."""
    assert x.ndim == 4
    batch, channels, in_h, in_w = x.shape

    # Upsample by inserting zeros.
    x = x.reshape(batch, channels, in_h, 1, in_w, 1)
    x = nn.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape(batch, channels, in_h * upy, in_w * upx)

    # Pad or crop.
    x = nn.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :,
          max(-pady0, 0): x.shape[2] - max(-pady1, 0),
          max(-padx0, 0): x.shape[3] - max(-padx1, 0)]

    # Setup filter.
    if f is not None:
        f = f.float32()
        if gain != 1:
            f = f * (gain ** (f.ndim / 2))
        if flip_filter:
            f = f.flip(list(range(f.ndim)))
        assert f.ndim == 2, "Separable 1D filters not yet supported in Jittor backend"
        fh, fw = f.shape
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])
        f_2d = f.reshape(1, 1, fh, fw)
        x = nn.conv2d(x, f_2d)
        x = x.reshape(batch, channels, x.shape[2], x.shape[3])
    elif gain != 1:
        x = x * gain

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x

#----------------------------------------------------------------------------
# Convenience wrappers.
#----------------------------------------------------------------------------

def upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1):
    r"""Upsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a multiple of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.
    """
    upx, upy = _parse_scaling(up)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]
    return upfirdn2d(x, f, up=up, padding=p, gain=gain * upx * upy, flip_filter=flip_filter)

def downsample2d(x, f, down=2, padding=0, flip_filter=False, gain=1):
    r"""Downsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a fraction of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.
    """
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]
    return upfirdn2d(x, f, down=down, padding=p, flip_filter=flip_filter, gain=gain)
