import math
from PIL import Image
import jittor as jt
from jittor import nn

from src.utils import math_utils
import numpy as np
from gghead_jittor.constants import DEFAULT_INTRINSICS
from src.utils.dreifus_compat import Pose, Intrinsics, CameraCoordinateConvention, PoseType

import numpy as np

from typing import Tuple, Literal

class GaussianCameraPoseSampler:
    """
    Samples pitch and yaw from a Gaussian distribution and returns a camera pose.
    Camera is specified as looking at the origin.
    If horizontal and vertical stddev (specified in radians) are zero, gives a
    deterministic camera pose with yaw=horizontal_mean, pitch=vertical_mean.
    The coordinate system is specified with y-up, z-forward, x-left.
    Horizontal mean is the azimuthal angle (rotation around y axis) in radians,
    vertical mean is the polar angle (angle from the y axis) in radians.
    A point along the z-axis has azimuthal_angle=0, polar_angle=pi/2.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = GaussianCameraPoseSampler.sample(math.pi/2, math.pi/2, radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = jt.randn((batch_size, 1)) * horizontal_stddev + horizontal_mean
        v = jt.randn((batch_size, 1)) * vertical_stddev + vertical_mean
        v = jt.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = jt.arccos(1 - 2*v)

        camera_origins = jt.zeros((batch_size, 3))

        camera_origins[:, 0:1] = radius*jt.sin(phi) * jt.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*jt.sin(phi) * jt.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*jt.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)


class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, lookat_position, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = jt.randn((batch_size, 1)) * horizontal_stddev + horizontal_mean
        v = jt.randn((batch_size, 1)) * vertical_stddev + vertical_mean
        v = jt.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = jt.arccos(1 - 2*v)

        camera_origins = jt.zeros((batch_size, 3))

        camera_origins[:, 0:1] = radius*jt.sin(phi) * jt.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*jt.sin(phi) * jt.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*jt.cos(phi)

        forward_vectors = math_utils.normalize_vecs(lookat_position - camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)

class UniformCameraPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    pose is sampled from a uniform distribution with range +-[horizontal/vertical]_stddev.

    Example:
    For a batch of random camera poses looking at the origin with yaw sampled from [-pi/2, +pi/2] radians:

    cam2worlds = UniformCameraPoseSampler.sample(math.pi/2, math.pi/2, horizontal_stddev=math.pi/2, radius=1, batch_size=16)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = (jt.rand((batch_size, 1)) * 2 - 1) * horizontal_stddev + horizontal_mean
        v = (jt.rand((batch_size, 1)) * 2 - 1) * vertical_stddev + vertical_mean
        v = jt.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = jt.arccos(1 - 2*v)

        camera_origins = jt.zeros((batch_size, 3))

        camera_origins[:, 0:1] = radius*jt.sin(phi) * jt.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*jt.sin(phi) * jt.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*jt.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)    

def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = math_utils.normalize_vecs(forward_vector)
    up_vector = jt.array([0, 1, 0]).float32().expand_as(forward_vector)

    right_vector = -math_utils.normalize_vecs(jt.cross(up_vector, forward_vector, dim=-1))
    up_vector = math_utils.normalize_vecs(jt.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = jt.init.eye(4).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = jt.stack((right_vector, up_vector, forward_vector), dim=-1)

    translation_matrix = jt.init.eye(4).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world


def FOV_to_intrinsics(fov_degrees, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """

    focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
    intrinsics = jt.array([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]]).float32()
    return intrinsics

def rand_c2w(cam_pivot,cam_num, alpha_deg=None, beta_deg=None,):
    
    cam_radius = 2.7
    #---------------- rand cam -------------------------------
    if alpha_deg is not None and beta_deg is not None:
        h = math.radians(alpha_deg) + math.pi / 2
        v = math.radians(beta_deg) + math.pi / 2
    else:
        h = (
            2 * (jt.rand((cam_num, 1)) - 0.5 ) * np.pi / 3
            + np.pi / 2
        )
        v = (
            2 * (jt.rand((cam_num, 1)) - 0.5 ) * np.pi / 4
            + np.pi / 2
        )
    poses = LookAtPoseSampler.sample(
        h,
        v,
        cam_pivot,
        radius=cam_radius,
        horizontal_stddev=0,
        vertical_stddev=0,
        batch_size=cam_num,
    ).reshape(-1, 16)
    
    cs = jt.concat(
        [
            poses,
            jt.array(np.array(DEFAULT_INTRINSICS.flatten(), dtype=np.float32)).unsqueeze(0).repeat(
                poses.shape[0], 1
            ),
        ],
        1,
    )
    return cs

def c2cam(cs=None):
    from src.utils.jgaussian_compat import pose_to_rendercam
    jt.sync_all()
    gaussian_camera = []
    for c in cs:
        cam2world_matrix = c[:16].reshape(4, 4)
        c2w_np = cam2world_matrix.detach().numpy()
        intrinsics_matrix = np.array(DEFAULT_INTRINSICS).reshape(3, 3)
        cam_2_world_pose = Pose(
            c2w_np,
            pose_type=PoseType.CAM_2_WORLD,
            disable_rotation_check=True,
        )
        intrinsics = Intrinsics(intrinsics_matrix)
        intrinsics = intrinsics.rescale(512, inplace=False)
        # NOTE: pose_to_rendercam from jgaussian_compat (Jittor)
        gaussian_camera.append(pose_to_rendercam(
            cam_2_world_pose, intrinsics, 512, 512, device='cuda'
        ))
    return gaussian_camera




# except ImportError:
#     # ... (占位符定义) ...
#     class Poser:
#         def __init__(self, device): pass
#         def get_pose(self, coeffs):
#             return {"pose": np.eye(4).flatten()[:16], "intrinsics": np.eye(3).flatten()}
#     def define_net_recon(*args, **kwargs):
#         return torch.nn.Sequential(torch.nn.Linear(3*224*224, 257))


class FaceRecon:
    """
    Face reconstruction for camera parameter estimation.
    NOTE: Deep3DFaceRecon internally uses torch. We convert outputs to jittor.
    """
    def __init__(self, device='cuda'):
        import torch
        from Deep3DFaceRecon.pose import Poser
        from Deep3DFaceRecon.Recon_networks import define_net_recon

        self.torch = torch
        self.torch_device = torch.device(device)
        self.pose_estimator = Poser(self.torch_device)
        
        print("Loading 3D face reconstruction model...")
        init_path = 'third_party/Deep3DFaceRecon/checkpoints/init_model/resnet50-0676ba61.pth'
        load_path = 'third_party/Deep3DFaceRecon/checkpoints/pretrained/epoch_20.pth'

        self.net_recon = define_net_recon(net_recon='resnet50', use_last_fc=False, init_path=init_path)
        state_dict = torch.load(load_path, map_location=self.torch_device)
        self.net_recon.load_state_dict(state_dict['net_recon'])
        self.net_recon.eval()
        self.net_recon.to(self.torch_device)
        print("Model loaded.")

    
    @staticmethod
    def _split_coeff(coeffs):
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs, 'exp': exp_coeffs, 'tex': tex_coeffs,
            'angle': angles, 'gamma': gammas, 'trans': translations
        }

    def __call__(self, input_data):
        """
        Process a PIL image or jittor Var, return camera params as jittor Var.
        input_data: PIL.Image or jittor.Var (B, 3, H, W) in [0, 1]
        Returns: (numpy coeffs, jittor camera_params (B, 25))
        """
        torch = self.torch
        with torch.no_grad():
            # Convert input to torch tensor for Deep3DFaceRecon
            if isinstance(input_data, Image.Image):
                image_resized_pil = input_data.resize((224, 224))
                im_tensor = torch.tensor(np.array(image_resized_pil) / 255., dtype=torch.float32) \
                                 .permute(2, 0, 1).unsqueeze(0).to(self.torch_device)
            elif hasattr(input_data, 'numpy'):  # jittor Var
                np_data = input_data.numpy()
                im_tensor = torch.from_numpy(np_data).float().to(self.torch_device)
                if im_tensor.ndim != 4 or im_tensor.shape[1] != 3:
                    raise ValueError(f"Input shape must be (B, 3, H, W), got {im_tensor.shape}")
                im_tensor = torch.nn.functional.interpolate(im_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            else:
                raise TypeError(f"Unsupported input type: {type(input_data)}")

            # Run torch-based model
            output_coeffs = self.net_recon(im_tensor)
            output_coeffs = output_coeffs.detach().cpu().numpy()
        
        pred_coeffs_dict = self._split_coeff(output_coeffs)
        pose_dict = self.pose_estimator.get_pose(pred_coeffs_dict)
        pose = pose_dict["pose"]

        # Convert output to jittor
        intrinsics_np = np.array(pose_dict["intrinsics"], dtype=np.float32)
        pose_np = np.array(pose, dtype=np.float32)
        camera_params = jt.concat([
            jt.array(pose_np).reshape(-1, 16),
            jt.array(intrinsics_np).reshape(-1, 9)
        ], 1)

        return output_coeffs, camera_params

    def _get_single_coeff_dict(self, coeff_dict_batch, index):
        return {key: val[index:index+1] for key, val in coeff_dict_batch.items()}



# import torch
# import math
# from typing import Tuple, Literal

def get_angles_from_camera_params(camera_params, lookat_position):
    """
    Recover h, v angles from camera params for LookAtPoseSampler.
    camera_params: (B, 25) or (B, 4, 4) jittor Var
    lookat_position: (3,) or (B, 3) jittor Var or list
    Returns: (h, v) as jittor Vars
    """
    if camera_params.shape[-1] == 25:
        c2w_matrix = camera_params[:, :16].view(-1, 4, 4)
    elif camera_params.shape[-2:] == (4, 4):
        c2w_matrix = camera_params
    else:
        raise ValueError(f"Invalid shape: {camera_params.shape}")

    if not isinstance(lookat_position, jt.Var):
        lookat_position = jt.array(lookat_position).float32()
    if lookat_position.ndim == 1:
        lookat_position = lookat_position.unsqueeze(0)

    camera_position_abs = c2w_matrix[:, :3, 3]
    relative_position = camera_position_abs - lookat_position
    
    radius = jt.norm(relative_position, p=2, dim=-1)
    safe_radius = jt.ternary(radius < 1e-8, jt.ones_like(radius), radius)
    
    x_rel, y_rel, z_rel = relative_position[:, 0], relative_position[:, 1], relative_position[:, 2]
    
    phi = jt.arccos(y_rel / safe_radius)
    theta = math.pi - jt.arctan2(z_rel, x_rel)
    
    h = theta
    v_normalized = (1 - jt.cos(phi)) / 2
    v = v_normalized * math.pi
    
    return h, v



def get_c3(camera_params, delta_h, delta_v, device=None):
    cam_pivot = jt.array([0, 0.05, 0.2]).float32()
    h, v = get_angles_from_camera_params(camera_params, cam_pivot)
    
    intrinsics = camera_params[:, 16:]
    cam_radius = 2.7

    pose = LookAtPoseSampler.sample(h, v, cam_pivot,
                                    radius=cam_radius, horizontal_stddev=0, vertical_stddev=0, batch_size=1).reshape(-1, 16)
    camera_params = jt.concat([pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    pose = LookAtPoseSampler.sample(h+delta_h, v+delta_v, cam_pivot,
                                    radius=cam_radius, horizontal_stddev=0, vertical_stddev=0, batch_size=1).reshape(-1, 16)
    camera_params1 = jt.concat([pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    
    pose = LookAtPoseSampler.sample(h-delta_h, v-delta_v, cam_pivot,
                                    radius=cam_radius, horizontal_stddev=0, vertical_stddev=0, batch_size=1).reshape(-1, 16)
    camera_params2 = jt.concat([pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    
    camera_params = [camera_params, camera_params1, camera_params2]

    return camera_params

