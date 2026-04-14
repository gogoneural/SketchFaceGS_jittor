"""
Lightweight pure-numpy replacement for the ``dreifus`` library.

Provides drop-in replacements for:
  - dreifus.camera.PoseType
  - dreifus.camera.CameraCoordinateConvention
  - dreifus.matrix.Pose
  - dreifus.matrix.Intrinsics

No torch dependency — only numpy.
"""

import math
import enum
import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PoseType(enum.Enum):
    CAM_2_WORLD = "CAM_2_WORLD"
    WORLD_2_CAM = "WORLD_2_CAM"


class CameraCoordinateConvention(enum.Enum):
    OPEN_CV = "OPEN_CV"
    OPEN_GL = "OPEN_GL"


# Conversion matrix: OpenGL <-> OpenCV flips y and z axes.
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


# ---------------------------------------------------------------------------
# Pose  (numpy ndarray subclass with metadata)
# ---------------------------------------------------------------------------

class Pose(np.ndarray):
    """4x4 pose matrix with ``pose_type`` and ``camera_coordinate_convention``."""

    def __new__(cls,
                matrix_or_rotation=None,
                translation=None,
                pose_type: PoseType = PoseType.CAM_2_WORLD,
                camera_coordinate_convention: CameraCoordinateConvention = CameraCoordinateConvention.OPEN_CV,
                disable_rotation_check: bool = False):

        if matrix_or_rotation is None:
            mat = np.eye(4, dtype=np.float64)
        else:
            _m = matrix_or_rotation.numpy() if hasattr(matrix_or_rotation, 'numpy') else matrix_or_rotation
            mat = np.asarray(_m, dtype=np.float64)

        if mat.shape == (3, 3):
            full = np.eye(4, dtype=np.float64)
            full[:3, :3] = mat
            if translation is not None:
                _t = translation.numpy() if hasattr(translation, 'numpy') else translation
                full[:3, 3] = np.asarray(_t, dtype=np.float64)
            mat = full
        elif mat.shape == (4, 4):
            if translation is not None:
                mat = mat.copy()
                _t = translation.numpy() if hasattr(translation, 'numpy') else translation
                mat[:3, 3] = np.asarray(_t, dtype=np.float64)
        else:
            raise ValueError(f"Expected (3,3) or (4,4), got {mat.shape}")

        obj = mat.view(cls)
        obj.pose_type = pose_type
        obj.camera_coordinate_convention = camera_coordinate_convention
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.pose_type = getattr(obj, 'pose_type', PoseType.CAM_2_WORLD)
        self.camera_coordinate_convention = getattr(
            obj, 'camera_coordinate_convention', CameraCoordinateConvention.OPEN_CV)

    # -- helpers --

    def get_rotation_matrix(self) -> np.ndarray:
        return np.asarray(self[:3, :3])

    def get_translation(self) -> np.ndarray:
        return np.asarray(self[:3, 3])

    def change_pose_type(self, target: PoseType, inplace: bool = False) -> "Pose":
        if self.pose_type == target:
            out = self if inplace else self.copy().view(Pose)
            out.pose_type = target
            out.camera_coordinate_convention = self.camera_coordinate_convention
            return out
        # Invert: c2w <-> w2c
        mat = np.linalg.inv(np.asarray(self))
        if inplace:
            self[...] = mat
            self.pose_type = target
            return self
        out = Pose(mat, pose_type=target,
                   camera_coordinate_convention=self.camera_coordinate_convention)
        return out

    def change_camera_coordinate_convention(
            self,
            target: CameraCoordinateConvention,
            inplace: bool = False) -> "Pose":
        if self.camera_coordinate_convention == target:
            out = self if inplace else self.copy().view(Pose)
            out.pose_type = self.pose_type
            out.camera_coordinate_convention = target
            return out

        # Convert: apply flip to rotation columns (and adjust translation sign)
        # For c2w: new = old @ flip;  for w2c: new = flip @ old
        mat = np.asarray(self).copy()
        if self.pose_type == PoseType.CAM_2_WORLD:
            mat = mat @ _GL_TO_CV
        else:
            mat = _GL_TO_CV @ mat

        if inplace:
            self[...] = mat
            self.camera_coordinate_convention = target
            return self
        return Pose(mat, pose_type=self.pose_type,
                    camera_coordinate_convention=target)


# ---------------------------------------------------------------------------
# Intrinsics  (numpy ndarray subclass)
# ---------------------------------------------------------------------------

class Intrinsics(np.ndarray):
    """3x3 camera intrinsics matrix with convenience properties."""

    def __new__(cls, matrix):
        _m = matrix.numpy() if hasattr(matrix, 'numpy') else matrix
        mat = np.asarray(_m, dtype=np.float64).reshape(3, 3)
        return mat.view(cls)

    def __array_finalize__(self, obj):
        pass

    # -- properties --

    @property
    def fx(self) -> float:
        return float(self[0, 0])

    @property
    def fy(self) -> float:
        return float(self[1, 1])

    @property
    def cx(self) -> float:
        return float(self[0, 2])

    @property
    def cy(self) -> float:
        return float(self[1, 2])

    def get_fovx(self, img_w: int) -> float:
        return 2.0 * math.atan(img_w / (2.0 * self.fx))

    def get_fovy(self, img_h: int) -> float:
        return 2.0 * math.atan(img_h / (2.0 * self.fy))

    def rescale(self, target_size: int, inplace: bool = False) -> "Intrinsics":
        """Scale intrinsics assuming original is normalised (cx,cy ~ 0.5)."""
        mat = np.asarray(self) if inplace else np.asarray(self).copy()
        mat[0, :] *= target_size
        mat[1, :] *= target_size
        out = mat.view(Intrinsics) if inplace else Intrinsics(mat)
        return out
