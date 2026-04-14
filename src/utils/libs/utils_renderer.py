#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import math
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from diff_gaussian_rasterization_32d import GaussianRasterizationSettings as  GaussianRasterizationSettings_32d
# from diff_gaussian_rasterization_32d import GaussianRasterizer as GaussianRasterizer_32d
import pdb
NUM_CHANNELS = 3

def render_gaussian(gs_params, cam_matrix, device, cam_params=None, sh_degree=0, bg_color=None,idx = 0):
    # Build params
    batch_size = cam_matrix.shape[0]
    focal_x, focal_y, cam_size = cam_params['focal_x'], cam_params['focal_y'], cam_params['size']
    points,shs, opacities, scales, rotations = \
         gs_params['xyz'], gs_params["shs"],  gs_params["opacity"], gs_params["scaling"], gs_params['rotation'],#gs_params['cov3d']
    view_mat, proj_mat, cam_pos = build_camera_matrices(cam_matrix.to(device), focal_x, focal_y)
    bg_color = cam_matrix.new_zeros(batch_size, NUM_CHANNELS, dtype=torch.float32).to(device) if bg_color is None else bg_color
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # colors = sh2colors(gs_params['colors'], gs_params['xyz'], cam_pos,sh_degree)
    means2D = torch.zeros_like(points, dtype=points.dtype, requires_grad=True, device=device) + 0.
    # means3D = points.detach().to("cuda").clone().requires_grad_(True)
    try:
        means2D.retain_grad()
    except:
        pass
    # Run rendering
    all_rendered, all_radii, all_depth, all_masks = [], [], [], []
    for bid in range(batch_size):
        #.contiguous()
        raster_settings = GaussianRasterizationSettings(
            sh_degree=sh_degree, bg=bg_color[bid],
            image_height=cam_size[0], image_width=cam_size[1],
            tanfovx=1.0 / focal_x, tanfovy=1.0 / focal_y,
            viewmatrix=view_mat[bid].to(device).contiguous(), projmatrix=proj_mat[bid].to(device).contiguous(), campos=cam_pos[bid].to(device).contiguous(),
            scale_modifier=1.0, prefiltered=False, debug=False, antialiasing=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rendered, radii, depth = rasterizer(
            means3D=points[bid].to(device).contiguous(), means2D=means2D[bid].to(device).contiguous(),
            shs=shs[bid].to(device).contiguous(), colors_precomp=None,
            opacities=opacities[bid].to(device).contiguous(), scales=scales[bid].to(device).contiguous(),
            rotations=rotations[bid].to(device).contiguous(),
            cov3D_precomp=None
        )
        all_rendered.append(rendered)
        all_radii.append(radii)
        # all_depth.append(depth)
        # raster_settings = GaussianRasterizationSettings_32d(
        #     sh_degree=sh_degree, bg=bg_color[bid],
        #     image_height=cam_size[0], image_width=cam_size[1],
        #     tanfovx=1.0 / focal_x, tanfovy=1.0 / focal_y,
        #     viewmatrix=view_mat[bid].to(device).contiguous(), projmatrix=proj_mat[bid].to(device).contiguous(),
        #     campos=cam_pos[bid].to(device).contiguous(),
        #     scale_modifier=1.0, prefiltered=False, debug=False,
        # )
        # rasterizer = GaussianRasterizer_32d(raster_settings=raster_settings)
        # masks, radii, = rasterizer(
        #     means3D=points[bid].to(device).contiguous(), means2D=means2D[bid].to(device).contiguous(),
        #     shs=None, colors_precomp=masks[bid].to(device).contiguous(),
        #     opacities=opacities[bid].to(device).contiguous(), scales=None,
        #     rotations=None, cov3D_precomp=cov3d[bid].to(device).contiguous()
        # )
        # all_masks.append(masks)
    all_rendered = torch.stack(all_rendered, dim=0)
    all_radii = torch.stack(all_radii, dim=0)
    # all_depth = torch.stack(all_depth, dim=0)
    # all_masks = torch.stack(all_masks, dim=0)
    # all_rendered = all_rendered[0][None,...]
    # all_radii = all_radii[0][None,...]
    return {
        "images": all_rendered, "radii": all_radii, "viewspace_points": means2D,#'depth':  all_depth,'masks':all_masks
    }


def build_camera_matrices(cam_matrix, focal_x, focal_y):
    def get_projection_matrix(fov_x, fov_y, z_near=0.01, z_far=100, device='cpu'):
        K = torch.zeros(4, 4, device=device)
        z_sign = 1.0
        K[0, 0] = 1.0 / math.tan((fov_x / 2))
        K[1, 1] = 1.0 / math.tan((fov_y / 2))
        K[3, 2] = z_sign
        K[2, 2] = z_sign * z_far / (z_far - z_near)
        K[2, 3] = -(z_far * z_near) / (z_far - z_near)
        return K

    def get_world_to_view_matrix(transforms):
        assert transforms.shape[-2:] == (3, 4)
        viewmatrix = transforms.new_zeros(transforms.shape[0], 4, 4)
        for i in range(4):
            viewmatrix[:, i, i] = 1.0
        viewmatrix[:, :3, :3] = transforms[:, :3, :3]
        viewmatrix[:, 3, :3] = transforms[:, :3, 3]
        viewmatrix[:, :, :2] *= -1.0
        return viewmatrix

    def get_full_projection_matrix(viewmatrix, fov_x, fov_y):
        proj_matrix = get_projection_matrix(fov_x, fov_y, device=viewmatrix.device)
        full_proj_matrix = viewmatrix @ proj_matrix.transpose(0, 1)
        return full_proj_matrix
    #
    # theta_deg = 30  # 角度制
    # theta = theta_deg * torch.pi / 180
    #
    # R_y = torch.tensor([
    #     [math.cos(theta), 0, math.sin(theta)],
    #     [0, 1, 0],
    #     [-math.sin(theta), 0, math.cos(theta)]
    # ]).to(cam_matrix)
    # theta = theta*2
    # R_2y = torch.tensor([
    #     [math.cos(theta), 0, math.sin(theta)],
    #     [0, 1, 0],
    #     [-math.sin(theta), 0, math.cos(theta)]
    # ]).to(device)
    # # 应用旋转
    # cam_matrix[ :, :3, :3] = R_y @ cam_matrix[ :, :3, :3]
    # cam_matrix[ :, :3, [3]] = R_2y @ cam_matrix[ :, :3, [3]]
    fov_x = 2 * math.atan(1.0 / focal_x)
    fov_y = 2 * math.atan(1.0 / focal_y)
    # view_matrix = get_world_to_view_matrix(cam_matrix)
    # full_proj_matrix = get_full_projection_matrix(view_matrix, fov_x, fov_y)
    # cam_pos = cam_matrix[:, :3, 3]
    view_matrix = getWorld2View(cam_matrix).transpose(-2, -1)
    # view_matrix[:, :3, :3] = R_y @ view_matrix[:, :3, :3]
    # 生成投影矩阵
    projection_matrix = getProjectionMatrix(znear=0.01, zfar=100, fovX=fov_x, fovY=fov_y,
                                           ).transpose(0, 1).to(cam_matrix.device)

    # 计算完整的投影变换矩阵
    full_proj_matrix = (view_matrix @ projection_matrix).to(cam_matrix.device)

    # 相机中心位置
    cam_pos = view_matrix.inverse()[..., 3, :3]#????????改了
    # cam_pos = (cam_pos[..., None, :] @ R_y).squeeze(-2)
    # cam_pos = cam_matrix[:, :3, 3]

    return view_matrix, full_proj_matrix, cam_pos
def build_camera_matrices2(cam_matrix, focal_x, focal_y, idx):
    def get_projection_matrix(fov_x, fov_y, z_near=0.01, z_far=100, device='cpu'):
        K = torch.zeros(4, 4, device=device)
        z_sign = 1.0
        K[0, 0] = 1.0 / math.tan((fov_x / 2))
        K[1, 1] = 1.0 / math.tan((fov_y / 2))
        K[3, 2] = z_sign
        K[2, 2] = z_sign * z_far / (z_far - z_near)
        K[2, 3] = -(z_far * z_near) / (z_far - z_near)
        return K

    def get_world_to_view_matrix(transforms):
        assert transforms.shape[-2:] == (3, 4)
        viewmatrix = transforms.new_zeros(transforms.shape[0], 4, 4)
        for i in range(4):
            viewmatrix[:, i, i] = 1.0
        viewmatrix[:, :3, :3] = transforms[:, :3, :3]
        viewmatrix[:, 3, :3] = transforms[:, :3, 3]
        viewmatrix[:, :, :2] *= -1.0
        return viewmatrix

    def get_full_projection_matrix(viewmatrix, fov_x, fov_y):
        proj_matrix = get_projection_matrix(fov_x, fov_y, device=viewmatrix.device)
        full_proj_matrix = viewmatrix @ proj_matrix.transpose(0, 1)
        return full_proj_matrix

    theta_deg = float(idx) # 角度制
    theta = theta_deg * torch.pi / 180

    R_y = torch.tensor([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)]
    ]).to(cam_matrix)
    # theta = theta*2
    # R_2y = torch.tensor([
    #     [math.cos(theta), 0, math.sin(theta)],
    #     [0, 1, 0],
    #     [-math.sin(theta), 0, math.cos(theta)]
    # ]).to(device)
    # # 应用旋转
    # cam_matrix[ :, :3, :3] = R_y @ cam_matrix[ :, :3, :3]
    # cam_matrix[ :, :3, [3]] = R_2y @ cam_matrix[ :, :3, [3]]
    fov_x = 2 * math.atan(1.0 / focal_x)
    fov_y = 2 * math.atan(1.0 / focal_y)
    # view_matrix = get_world_to_view_matrix(cam_matrix)
    # full_proj_matrix = get_full_projection_matrix(view_matrix, fov_x, fov_y)
    # cam_pos = cam_matrix[:, :3, 3]
    view_matrix = getWorld2View(cam_matrix).transpose(-2, -1)
    view_matrix[:, :3, :3] = R_y @ view_matrix[:, :3, :3]
    # 生成投影矩阵
    projection_matrix = getProjectionMatrix(znear=0.01, zfar=100, fovX=fov_x, fovY=fov_y,
                                           ).transpose(0, 1).to(cam_matrix.device)

    # 计算完整的投影变换矩阵
    full_proj_matrix = (view_matrix @ projection_matrix).to(cam_matrix.device)

    # 相机中心位置
    cam_pos = view_matrix.inverse()[..., 3, :3]#????????改了
    cam_pos = (cam_pos[..., None, :] @ R_y).squeeze(-2)
    # cam_pos = cam_matrix[:, :3, 3]

    return view_matrix, full_proj_matrix, cam_pos
def get_local_pos_from_uv(gs_depth,gs_uv,points,focal_x,focal_y,cam_matrix,n_points,orientation):
    fov_x = 2 * math.atan(1.0 / focal_x)
    fov_y = 2 * math.atan(1.0 / focal_y)
    projection_matrix = getProjectionMatrix(znear=0.01, zfar=100, fovX=fov_x, fovY=fov_y,
                                            )
    gs_xy = gs_uv*gs_depth
    gs_xy[...,0] = gs_xy[...,0]/(projection_matrix[0,0].item())
    gs_xy[..., 1] = gs_xy[..., 1] / (projection_matrix[1, 1].item())
    view_matrix, full_proj_matrix, cam_pos = build_camera_matrices(cam_matrix, focal_x, focal_y)
    c2w = view_matrix.inverse()
    cam_points_homogeneous = torch.cat(
        (gs_xy, gs_depth,torch.ones(gs_xy.shape[0], gs_xy.shape[1], 1).to(gs_xy.device)),
        dim=-1).unsqueeze(-2)# [B,N, 1,4]
    world_points = (cam_points_homogeneous@(c2w.unsqueeze(1)) ).squeeze(-2)
    world_points_off = world_points.reshape(world_points.shape[0],-1,n_points,4)[...,:3] - points[...,None,:]
    local_pos_off = world_points_off@(orientation.inverse())
    return local_pos_off
def world_to_screen(world_points, cam_matrix, focal_x, focal_y):
    # 1. 构建相机矩阵
    view_matrix, full_proj_matrix, cam_pos = build_camera_matrices(cam_matrix, focal_x, focal_y)

    # 2. 将世界坐标转换为相机坐标
    world_points_homogeneous = torch.cat((world_points, torch.ones(world_points.shape[0], world_points.shape[1], 1).to(world_points.device)), dim=-1).unsqueeze(-2)   # [B,N, 1,4]

    camera_points = (world_points_homogeneous@(view_matrix.unsqueeze(1)) ).squeeze(-2)  # [B, N, 4]

    camera_points_homogeneous = camera_points.unsqueeze(-2)
    # 3. 使用投影矩阵将相机坐标转换为屏幕坐标

    projected_points = (world_points_homogeneous @ (full_proj_matrix.unsqueeze(1) )).squeeze(-2)  # [B, N, 4]

    # 4. 进行齐次除法，转换为非齐次坐标
    projected_points /= projected_points[..., [-1]] # 归一化到屏幕坐标
    depth = camera_points[..., [2]]
    dir = world_points - cam_pos[:,None,:]
    # 返回投影到屏幕上的坐标
    return projected_points[..., :2], depth,dir  # 返回u, v坐标

def getWorld2View(cam_matrix):
    R = cam_matrix[...,:3,:3]
    t = cam_matrix[...,:3, 3]
    Rt = torch.zeros((cam_matrix.shape[0], 4, 4))
    Rt[..., :3, :3] = R.transpose(-1,-2)
    Rt[..., :3, 3] = t
    Rt[..., 3, 3] = 1.0
    Rt[..., :2, :] *= -1.0
    return Rt.to(cam_matrix.device)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def sh2colors(shs,xyz,viewpoint_camera,deg):
    shs_view = shs.view(shs.shape[0],shs.shape[1],3,-1).contiguous()
    dir_pp = (xyz - viewpoint_camera[:,None,:].repeat(1,xyz.shape[1], 1))
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
    sh2rgb = eval_sh(deg, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    return colors_precomp


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]
def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result