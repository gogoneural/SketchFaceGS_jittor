"""
Compatibility shim: provides gaussian_splatting-equivalent interfaces
using JGaussian (Jittor) as the backend.

Replaces:
  - gaussian_splatting.arguments.PipelineParams2
  - gaussian_splatting.scene.cameras.pose_to_rendercam  (-> RenderCam)
"""

import math
import numpy as np
from dataclasses import dataclass

import jittor as jt
from src.utils.dreifus_compat import Pose, Intrinsics, CameraCoordinateConvention, PoseType

from utils.graphics_utils import getWorld2View2, getProjectionMatrix


# ---------------------------------------------------------------------------
# PipelineParams2  (drop-in replacement)
# ---------------------------------------------------------------------------
@dataclass
class PipelineParams2:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False


# ---------------------------------------------------------------------------
# RenderCam  (Jittor version — mirrors JGaussian Camera interface)
# ---------------------------------------------------------------------------
class RenderCam:
    """Lightweight camera object accepted by JGaussian's ``render()``.

    Exposes the same attributes as JGaussian's ``scene.cameras.Camera``:
      - image_width, image_height
      - FoVx, FoVy
      - world_view_transform   (4x4 jt.Var)
      - full_proj_transform    (4x4 jt.Var)
      - camera_center          (3,  jt.Var)
    """

    def __init__(self, width, height, R, T, FoVx, FoVy,
                 znear=0.01, zfar=100.0,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        self.image_width = width
        self.image_height = height
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.R = R
        self.T = T

        self.world_view_transform = jt.array(
            getWorld2View2(R, T, trans, scale)
        ).transpose(0, 1)

        projection_matrix = getProjectionMatrix(
            znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy
        ).transpose(0, 1)

        self.full_proj_transform = jt.nn.bmm(
            self.world_view_transform.unsqueeze(0),
            projection_matrix.unsqueeze(0),
        ).squeeze(0)

        self.camera_center = jt.linalg.inv(
            self.world_view_transform
        )[3, :3]


# ---------------------------------------------------------------------------
# pose_to_rendercam  (Jittor version)
# ---------------------------------------------------------------------------
def pose_to_rendercam(pose: Pose,
                      intrinsics: Intrinsics,
                      img_w: int,
                      img_h: int,
                      znear: float = 0.01,
                      zfar: float = 100.0,
                      device: str = 'cuda') -> RenderCam:
    """Convert a dreifus ``Pose`` + ``Intrinsics`` into a ``RenderCam``.

    This is the Jittor equivalent of
    ``gaussian_splatting.scene.cameras.pose_to_rendercam``.
    """
    fov_x = intrinsics.get_fovx(img_w)
    fov_y = intrinsics.get_fovy(img_h)

    pose = pose.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)
    pose = pose.change_camera_coordinate_convention(
        CameraCoordinateConvention.OPEN_CV, inplace=False
    )
    pose = pose.change_pose_type(PoseType.WORLD_2_CAM, inplace=False)

    T = pose.get_translation()
    R = pose.get_rotation_matrix().transpose()

    return RenderCam(img_w, img_h, R, T, fov_x, fov_y,
                     znear=znear, zfar=zfar)
