"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
Modified from smplx code for FLAME by Xuangeng Chu (xg.chu@outlook.com)
"""
import os

import pickle
import numpy as np
import jittor as jt
from jittor import nn

from .lbs import lbs, batch_rodrigues, vertices2landmarks


def _to_jt(x):
    """Convert numpy array to jittor Var."""
    if isinstance(x, np.ndarray):
        return jt.array(x)
    return x


class FLAMEModel(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """
    def __init__(self, n_shape, n_exp, scale=1.0, no_lmks=False):
        super().__init__()
        self.scale = scale
        self.no_lmks = no_lmks
        # print("creating the FLAME Model")
        _abs_path = os.path.dirname(os.path.abspath(__file__))
        self.flame_path = os.path.join(_abs_path, '../../../../assets')
        # Load from pre-exported numpy pickle (use scripts/export_flame_weights.py)
        with open(os.path.join(self.flame_path, 'FLAME_with_eye_numpy.pkl'), 'rb') as _f:
            flame_ckpt = pickle.load(_f)
        flame_model = flame_ckpt['flame_model']
        flame_lmk = flame_ckpt['lmk_embeddings']
        
        self.dtype = 'float32'
        self.faces_tensor = _to_jt(flame_model['f']).stop_grad()
        self.v_template = _to_jt(flame_model['v_template']).stop_grad()
        shapedirs = flame_model['shapedirs']
        self.shapedirs = _to_jt(np.concatenate([shapedirs[:, :, :n_shape], shapedirs[:, :, 300:300 + n_exp]], axis=2)).stop_grad()
        num_pose_basis = flame_model['posedirs'].shape[-1]
        self.posedirs = _to_jt(flame_model['posedirs'].reshape(-1, num_pose_basis).T).stop_grad()
        self.J_regressor = _to_jt(flame_model['J_regressor']).stop_grad()
        parents = flame_model['kintree_table'][0].copy()
        parents[0] = -1
        self.parents = _to_jt(parents).stop_grad()
        self.lbs_weights = _to_jt(flame_model['weights']).stop_grad()
        # Fixing Eyeball and neck rotation
        self.eye_pose = jt.zeros([1, 6]).stop_grad()
        self.neck_pose = jt.zeros([1, 3]).stop_grad()

        # Static and Dynamic Landmark embeddings for FLAME
        self.lmk_faces_idx = _to_jt(flame_lmk['static_lmk_faces_idx']).stop_grad()
        self.lmk_bary_coords = _to_jt(flame_lmk['static_lmk_bary_coords'].astype(np.float32)).stop_grad()
        self.dynamic_lmk_faces_idx = _to_jt(flame_lmk['dynamic_lmk_faces_idx'].astype(np.int64)).stop_grad()
        self.dynamic_lmk_bary_coords = _to_jt(flame_lmk['dynamic_lmk_bary_coords'].astype(np.float32)).stop_grad()
        self.full_lmk_faces_idx = _to_jt(flame_lmk['full_lmk_faces_idx_with_eye'].astype(np.int64)).stop_grad()
        self.full_lmk_bary_coords = _to_jt(flame_lmk['full_lmk_bary_coords_with_eye'].astype(np.float32)).stop_grad()

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = NECK_IDX
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = int(self.parents[curr_idx].item())
        self.neck_kin_chain = jt.array(neck_kin_chain, dtype='int32').stop_grad()
        # print("FLAME Model Done.")

    def get_faces(self, ):
        return self.faces_tensor.long()

    def _find_dynamic_lmk_idx_and_bcoords(
            self, pose, dynamic_lmk_faces_idx, dynamic_lmk_b_coords,
            neck_kin_chain, dtype='float32'
        ):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = pose.view(batch_size, -1, 3)[:, neck_kin_chain]
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = jt.init.eye(3, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = jt.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = jt.round(
            jt.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max_v=39)
        ).int32()

        neg_mask = (y_rot_angle < 0).int32()
        mask = (y_rot_angle < -39).int32()
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = dynamic_lmk_faces_idx[y_rot_angle]
        dyn_lmk_b_coords = dynamic_lmk_b_coords[y_rot_angle]
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def execute(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None, verts_sclae=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        if pose_params is None:
            pose_params = self.eye_pose.expand(batch_size, -1)
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        if expression_params is None:
            expression_params = jt.zeros(batch_size, self.cfg.n_exp)

        betas = jt.concat([shape_params, expression_params], dim=1)
        full_pose = jt.concat([
                pose_params[:, :3], self.neck_pose.expand(batch_size, -1), 
                pose_params[:, 3:], eye_pose_params
            ], dim=1
        )
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, _ = lbs(
            betas, full_pose, template_vertices,
            self.shapedirs, self.posedirs, self.J_regressor, self.parents,
            self.lbs_weights, dtype=self.dtype, detach_pose_correctives=False
        )
        if self.no_lmks:
            return vertices * self.scale
        landmarks3d = vertices2landmarks(
            vertices, self.faces_tensor, 
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1)
        )
        landmark_3d = reselect_eyes(vertices, landmarks3d)
        if verts_sclae is not None:
            return vertices * verts_sclae, landmark_3d * verts_sclae
        return vertices * self.scale, landmarks3d * self.scale

    def forward_withoffset(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None,
                verts_sclae=None,offset=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        if pose_params is None:
            pose_params = self.eye_pose.expand(batch_size, -1)
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        if expression_params is None:
            expression_params = jt.zeros(batch_size, self.cfg.n_exp)

        betas = jt.concat([shape_params, expression_params], dim=1)
        full_pose = jt.concat([
            pose_params[:, :3], self.neck_pose.expand(batch_size, -1),
            pose_params[:, 3:], eye_pose_params
        ], dim=1
        )
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1) + offset
        vertices, _ = lbs(
            betas, full_pose, template_vertices,
            self.shapedirs, self.posedirs, self.J_regressor, self.parents,
            self.lbs_weights, dtype=self.dtype, detach_pose_correctives=False
        )
        if self.no_lmks:
            return vertices * self.scale
        landmarks3d = vertices2landmarks(
            vertices, self.faces_tensor,
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1)
        )
        landmark_3d = reselect_eyes(vertices, landmarks3d)
        if verts_sclae is not None:
            return vertices * verts_sclae, landmark_3d * verts_sclae
        return vertices * self.scale, landmarks3d * self.scale
    def _vertices2landmarks(self, vertices):
        landmarks3d = vertices2landmarks(
            vertices, self.faces_tensor, 
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1)
        )
        landmark_3d = reselect_eyes(vertices, landmarks3d)
        return landmark_3d


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = jt.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return jt.arctan2(-rot_mats[:, 2, 0], sy)


def reselect_eyes(vertices, lmks70):
    lmks70 = lmks70.clone()
    eye_in_shape = [2422,2422, 2452, 2454, 2471, 3638, 2276, 2360, 3835, 1292, 1217, 1146, 1146, 999, 827, ]
    eye_in_shape_reduce = [0,2,4,5,6,7,8,9,10,11,13,14]
    cur_eye = vertices[:, eye_in_shape]
    cur_eye[:, 0] = (cur_eye[:, 0] + cur_eye[:, 1]) * 0.5
    cur_eye[:, 2] = (cur_eye[:, 2] + cur_eye[:, 3]) * 0.5
    cur_eye[:, 11] = (cur_eye[:, 11] + cur_eye[:, 12]) * 0.5
    cur_eye = cur_eye[:, eye_in_shape_reduce]
    lmks70[:, [37,38,40,41,43,44,46,47]] = cur_eye[:, [1,2,4,5,7,8,10,11]]
    return lmks70
