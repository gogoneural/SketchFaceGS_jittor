import numpy as np
import torch
import trimesh
from argparse import ArgumentParser

# calculated using EG3D
_x = torch.Tensor([
    3.774520955187813526e-02,
    -3.500493153554742887e-02,
    -5.264882553835513283e-03,
    6.340316100444725808e-04,
    2.257675357421122173e-02,
    1.039790552171645927e-01,
    2.514746581238903822e+00,
])

# calculated using panohead
_x_panohead = torch.Tensor([
    -1.381084599590920838e-03,
    -3.560132731382278926e-02,
    1.066695961042541350e-02,
    4.289061343825034461e-04,
    -1.509171220091529500e-02,
    5.460469403427847118e-02,
    2.285124474978020626e+00,
])


def get_RT(alpha, beta, gamma, Tx, Ty, Tz, s):
    sa = torch.sin(alpha)
    ca = torch.cos(alpha)
    sb = torch.sin(beta)
    cb = torch.cos(beta)
    sg = torch.sin(gamma)
    cg = torch.cos(gamma)
    RT = torch.Tensor([
        [cb*cg, sa*sb*cg-ca*sg, ca*sb*cg+sa*sg, Tx],
        [cb*sg, sa*sb*sg+ca*cg, ca*sb*sg-sa*cg, Ty],
        [-sb,   sa*cb,          ca*cb,          Tz],
        [0,     0,              0,              1],
    ])
    scale = torch.Tensor([
        [s, 0, 0, 0],
        [0, s, 0, 0], 
        [0, 0, s, 0], 
        [0, 0, 0, 1], 
    ])
    return RT, scale

_RT, _scale = get_RT(_x[0], _x[1], _x[2], _x[3], _x[4], _x[5], _x[6])

def align_verts(verts, RT=_RT, scale=_scale):
    RT = RT.to(verts.device)
    scale = scale.to(verts.device)
    # print(scale.shape, verts.shape, RT.shape, RT[0:3,0:3].shape, RT[0:3,[3]].shape)
    # x = verts.matmul(RT[0:3,0:3].T)
    # x = x * scale
    # x = x + RT[0:3,[3]].T
    return verts.matmul(RT[0:3,0:3].T).matmul(scale[0:3, 0:3]) + RT[0:3,[3]].T

def mesh_normals(verts, faces):
    face_verts = verts[..., faces, :]
    _shape = face_verts.shape
    a, b, c = face_verts[..., 0, :], face_verts[..., 1, :], face_verts[..., 2, :]
    normals = torch.cross(b - a, c - a, dim=-1)
    verts_normals = torch.zeros_like(verts)
    verts_normals.index_add_(-2, faces.view(-1), normals.view(*_shape[:-2], 1, 3).expand(*_shape[:-2], 3, 3).reshape(*_shape[:-3], -1, 3))
    verts_normals = verts_normals / (torch.norm(verts_normals, dim=-1, keepdim=True))
    return verts_normals

def flame_postprocess(v, f):
    v = align_verts(v)
    vn = mesh_normals(v, f)
    v = torch.cat([v, vn], -1)
    return v, f

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    # x = np.loadtxt("EG3D_RT_mean_all.txt")
    # x = torch.Tensor(x)
    x = _xx
    pred_RT, pred_scale = get_RT(x[0], x[1], x[2], x[3], x[4], x[5], x[6])

    ori_mesh = trimesh.load(args.input)
    ori_mesh.vertices = x[6] * (ori_mesh.vertices.dot(pred_RT[0:3,0:3].T) )+ pred_RT[0:3,3].T 
    ori_mesh.export(args.output)