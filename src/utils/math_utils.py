# MIT License

# Copyright (c) 2022 Petr Kellnhofer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import jittor as jt

def transform_vectors(matrix, vectors4):
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = jt.matmul(vectors4, matrix.transpose(-2, -1))
    return res


def normalize_vecs(vectors):
    """
    Normalize vector lengths.
    """
    return vectors / (jt.norm(vectors, dim=-1, keepdim=True))

def torch_dot(x, y):
    """
    Dot product of two tensors.
    """
    return (x * y).sum(-1)


def get_ray_limits_box(rays_o, rays_d, box_side_length):
    """
    Author: Petr Kellnhofer
    Intersects rays with the [-1, 1] NDC volume.
    Returns min and max distance of entry.
    Returns -1 for no intersection.
    """
    o_shape = rays_o.shape
    rays_o = rays_o.detach().reshape(-1, 3)
    rays_d = rays_d.detach().reshape(-1, 3)

    bb_min = [-1*(box_side_length/2), -1*(box_side_length/2), -1*(box_side_length/2)]
    bb_max = [1*(box_side_length/2), 1*(box_side_length/2), 1*(box_side_length/2)]
    bounds = jt.array([bb_min, bb_max]).float32()
    is_valid = jt.ones(rays_o.shape[:-1]).bool()

    # Precompute inverse for stability.
    invdir = 1 / rays_d
    sign = (invdir < 0).int32()

    # Intersect with YZ plane.
    tmin = (bounds[sign[..., 0]][..., 0] - rays_o[..., 0]) * invdir[..., 0]
    tmax = (bounds[1 - sign[..., 0]][..., 0] - rays_o[..., 0]) * invdir[..., 0]

    # Intersect with XZ plane.
    tymin = (bounds[sign[..., 1]][..., 1] - rays_o[..., 1]) * invdir[..., 1]
    tymax = (bounds[1 - sign[..., 1]][..., 1] - rays_o[..., 1]) * invdir[..., 1]

    # Resolve parallel rays.
    is_valid = is_valid & ~((tmin > tymax) | (tymin > tmax))

    # Use the shortest intersection.
    tmin = jt.maximum(tmin, tymin)
    tmax = jt.minimum(tmax, tymax)

    # Intersect with XY plane.
    tzmin = (bounds[sign[..., 2]][..., 2] - rays_o[..., 2]) * invdir[..., 2]
    tzmax = (bounds[1 - sign[..., 2]][..., 2] - rays_o[..., 2]) * invdir[..., 2]

    # Resolve parallel rays.
    is_valid = is_valid & ~((tmin > tzmax) | (tzmin > tmax))

    # Use the shortest intersection.
    tmin = jt.maximum(tmin, tzmin)
    tmax = jt.minimum(tmax, tzmax)

    # Mark invalid.
    tmin = jt.ternary(is_valid, tmin, jt.float32(-1))
    tmax = jt.ternary(is_valid, tmax, jt.float32(-2))

    return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1)


def linspace(start, stop, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates the multi-dimensional behaviour of numpy.linspace.
    """
    steps = jt.arange(num).float32() / (num - 1)

    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    out = start[None] + steps * (stop - start)[None]

    return out
