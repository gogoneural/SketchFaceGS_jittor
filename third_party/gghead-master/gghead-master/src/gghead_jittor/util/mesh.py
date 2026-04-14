from typing import Optional

import numpy as np
import jittor as jt
import trimesh
from tqdm import tqdm
try:
    import pyvista as pv
except (ImportError, OSError) as e:
    # Sometimes vtk has this weird bug "Error loading vtkioss-9.3-c0f4add4a2b52302512f2df0aa56b1e8.dll; The operation completed successfully." ...
    print(e)
    pv = None

try:
    from utils.general_utils import build_scaling_rotation, build_rotation
except ImportError:
    # Inline fallback if JGaussian is not on sys.path
    def build_rotation(r):
        q = r / jt.norm(r, dim=-1, keepdim=True)
        R = jt.zeros((q.shape[0], 3, 3))
        r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return R

    def build_scaling_rotation(s, r):
        L = jt.zeros((s.shape[0], 3, 3))
        R = build_rotation(r)
        L[:, 0, 0] = s[:, 0]
        L[:, 1, 1] = s[:, 1]
        L[:, 2, 2] = s[:, 2]
        return R @ L


def gaussians_to_mesh(
        gaussian_positions: jt.Var,
        gaussian_scales: jt.Var,
        gaussian_rotations: jt.Var,
        gaussian_colors: jt.Var,
        gaussian_opacities: jt.Var,
        use_spheres: bool = True,
        random_colors: bool = False,
        scale_factor: float = 1.5,
        ellipsoid_res: int = 5,
        opacity_threshold: float = 0.01,
        max_n_gaussians: Optional[int] = None,
        include_alphas: bool = False
) -> trimesh.Trimesh:
    gaussian_positions = gaussian_positions.detach().numpy()
    gaussian_colors = gaussian_colors.detach().numpy()
    gaussian_opacities = gaussian_opacities.detach().numpy()

    n_gaussians = len(gaussian_positions) if max_n_gaussians is None else max_n_gaussians

    if use_spheres:
        points = []
        faces = []
        points_count = 0
        face_count = 0
        all_vertex_colors = []

        base = trimesh.creation.icosphere(subdivisions=1)  # radius=0.5, count=16)

        rotm = build_scaling_rotation(gaussian_scales * scale_factor, gaussian_rotations).numpy()
        for i in range(n_gaussians):
            if gaussian_opacities[i] >= opacity_threshold:
                points.append(base.vertices @ rotm[i, ...].T + gaussian_positions[i:i + 1, :])
                tris = base.faces
                face_count += tris.shape[0]
                faces.append(tris + points_count)
                points_count += base.vertices.shape[0]

                if random_colors:
                    sphere_color = np.random.rand(3)
                else:
                    sphere_color = gaussian_colors[i]
                if include_alphas:
                    vertex_colors = np.tile(np.concatenate([sphere_color[None, :], np.clip(gaussian_opacities[[i]], 0, 1)], axis=1), [base.vertices.shape[0], 1])
                else:
                    vertex_colors = np.tile(sphere_color[None, :], [base.vertices.shape[0], 1])
                all_vertex_colors.append(vertex_colors)

        points = np.concatenate(points, axis=0)
        all_vertex_colors = np.concatenate(all_vertex_colors, axis=0)
        faces = np.concatenate(faces, axis=0)
        combined_mesh = trimesh.Trimesh(points, faces, process=False, vertex_colors=all_vertex_colors)

    else:
        if pv is None:
            raise ImportError("pyvista is required when use_spheres=False")
        gaussian_scales = gaussian_scales.numpy()
        gaussian_rotations = build_rotation(gaussian_rotations).numpy()

        ellipsoids = []
        for i in tqdm(list(range(n_gaussians))):
            scale = gaussian_scales[i] * scale_factor
            ellipsoid = pv.ParametricEllipsoid(scale[0], scale[1], scale[2], center=gaussian_positions[i], u_res=ellipsoid_res, v_res=ellipsoid_res,
                                               w_res=ellipsoid_res)
            ellipsoids.append(ellipsoid)

        all_vertex_colors = []
        ellipsoid_meshes = []
        for ellipsoid, ellipsoid_center, ellipsoid_color, ellipsoid_opacity, ellipsoid_rotation in zip(ellipsoids, gaussian_positions, gaussian_colors,
                                                                                                       gaussian_opacities, gaussian_rotations):
            if ellipsoid_opacity >= opacity_threshold:
                faces_as_array = ellipsoid.faces.reshape((ellipsoid.n_cells, 4))[:, 1:]
                # tmesh = trimesh.Trimesh(ellipsoid.points, faces_as_array, process=False, vertex_colors=np.concatenate([ellipsoid_color, ellipsoid_opacity]))
                vertices = ellipsoid.points
                vertices = ((vertices - ellipsoid_center) @ ellipsoid_rotation) + ellipsoid_center
                if random_colors:
                    ellipsoid_color = np.random.rand(3)
                tmesh = trimesh.Trimesh(vertices, faces_as_array, process=False, vertex_colors=ellipsoid_color)
                all_vertex_colors.extend(tmesh.visual.vertex_colors)
                ellipsoid_meshes.append(tmesh)
        combined_mesh = trimesh.util.concatenate(ellipsoid_meshes)

    return combined_mesh
