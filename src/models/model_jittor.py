#!/usr/bin/env python3
import os
import sys
import numpy as np
import time
import random
from dataclasses import dataclass
from typing import Tuple
import cv2
from scipy.ndimage import binary_dilation

# [Jittor 修改] 替换 PyTorch 核心库
import jittor as jt
from jittor import nn
from jittor import transform

# -----------------------------------------------------------------------------
# 第三方库依赖
# [✅已转换] gaussian_splatting → JGaussian (Jittor 原生)
# [✅已转换] gghead → gghead_jittor
# [✅已转换] dreifus → dreifus_compat (纯 numpy，无 torch 依赖)
# [✅已转换] 所有内部项目模块已完成 Jittor 转换
# -----------------------------------------------------------------------------

# Core utilities — 纯 numpy 实现，替代 dreifus
from src.utils.dreifus_compat import CameraCoordinateConvention, PoseType, Intrinsics, Pose

# GGHead — 外部库 (Jittor version)
import pickle
from gghead_jittor.constants import DEFAULT_INTRINSICS
from gghead_jittor.models.gghead_model import GGHeadModel, GGHeadConfig

# LHM (local third_party) [✅已转换为 Jittor]
from LHM.models.encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper
from LHM.models.transformer import TransformerDecoder
from LHM.models.rendering.gs_renderer import GSLayer
from LHM.models.rendering.utils.utils import MLP

# [✅已转换] 3D Gaussian Splatting → JGaussian (Jittor)
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.sh_utils import C0, eval_sh
from src.utils.jgaussian_compat import PipelineParams2, pose_to_rendercam

# Project internal modules [✅已转换为 Jittor]
from src.models.modules.gfpganv1_clean_arch import GFPGANv1Clean
from src.models.modules.networks_maskGAN import weights_init, GlobalGenerator_adain
from src.models.modules.conv import ZeroConv
from src.models.libs.flame_model import FLAMEModel

from src.utils.camera_utils import c2cam, rand_c2w
from src.utils.gaussian_utils import clone_gaussian_model
from src.utils.sketch_utils import random_get_sketch
from src.utils.video_utils import get_cs_list
from ..utils.gaussian_utils import _apply_opacity_activation, _apply_color_activation


@dataclass
class ImageMetadata:
    participant_id: int
    sequence_name: str
    timestep: int
    serial: str


def decode_camera_params(camera_params: np.ndarray, disable_rotation_check: bool = False) -> Tuple[Pose, Intrinsics]:
    pose = Pose(
        camera_params[:16].reshape((4, 4)),
        pose_type=PoseType.CAM_2_WORLD,
        disable_rotation_check=disable_rotation_check,
    )
    intrinsics = Intrinsics(camera_params[16:].reshape((3, 3)))
    return pose, intrinsics


def encode_camera_params(pose: Pose, intrinsics: Intrinsics) -> np.ndarray:
    pose = pose.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)
    pose = pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV, inplace=False)
    return np.concatenate([pose.flatten(), intrinsics.flatten()])

# -----------------------------------------------------------------------------
# 5. Global Configuration & Initialization
# -----------------------------------------------------------------------------

# [Jittor 修改] 数据增强替换为 Jittor transform
transform_pipeline = transform.Compose([
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("Initializing FLAME model...")
# [✅已转换] FLAMEModel 已完成 Jittor 转换
flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=5, no_lmks=True)
faces = flame_model.get_faces()
i0 = faces[..., 0]
i1 = faces[..., 1]
i2 = faces[..., 2]

# ================================================================
# Helper Classes (dependencies for SketchFaceGS)
# ================================================================

def random_irregular_mask_smooth(height: int, width: int, min_area: int = None, max_area: int = None,
                                 connectivity: int = 4, seed: int = None, blur_ksize: int = 15,
                                 blur_sigma: float = 5.0, thr: float = 0.5, area: list = [20, 3]) -> jt.Var:
    # 纯 NumPy 和 OpenCV 处理，无需大改
    if seed is not None:
        np.random.seed(seed)

    total = height * width
    if min_area is None:
        min_area = total // area[0]
    if max_area is None:
        max_area = total // area[1]
    target_size = np.random.randint(min_area, max_area + 1)

    start = (np.random.randint(height), np.random.randint(width))
    region = {start}
    frontier = [start]

    if connectivity == 8:
        neighs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        neighs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while frontier and len(region) < target_size:
        idx = np.random.randint(len(frontier))
        r, c = frontier.pop(idx)
        np.random.shuffle(neighs)
        for dr, dc in neighs:
            nbr = (r + dr, c + dc)
            if 0 <= nbr[0] < height and 0 <= nbr[1] < width and nbr not in region:
                region.add(nbr)
                frontier.append(nbr)
                if len(region) >= target_size:
                    break

    mask0 = np.zeros((height, width), dtype=np.float32)
    for r, c in region:
        mask0[r, c] = 1.0

    mask_uint8 = (mask0 * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(mask_uint8, (blur_ksize, blur_ksize), blur_sigma)
    _, mask_thr = cv2.threshold(blur, int(thr * 255), 255, cv2.THRESH_BINARY)
    mask_smooth = mask_thr.astype(np.float32) / 255.0

    # [Jittor 修改] torch.from_numpy -> jt.array
    return jt.array(mask_smooth).unsqueeze(0)


def dilate_mask_scipy(mask: jt.Var, n: int) -> jt.Var:
    """GPU-based mask dilation using max_pool2d (odd kernel required for same-size output)."""
    inp = mask.float() if mask.dtype != jt.float32 else mask
    if inp.ndim == 2:
        inp = inp.unsqueeze(0).unsqueeze(0)
    elif inp.ndim == 3:
        inp = inp.unsqueeze(0)
    ksize = n if n % 2 == 1 else n + 1  # max_pool2d same-padding requires odd kernel
    dilated = jt.nn.pool(inp, kernel_size=ksize, stride=1, padding=ksize // 2, op='maximum')
    return dilated.bool()


class Feature_Transformer_Extract(nn.Module):
    def __init__(self, dim_emb=512, transformer_layers=4, transformer_heads=16, mlp_layers=1, idxim=None, barim=None):
        super(Feature_Transformer_Extract, self).__init__()
        # [✅已转换] Dinov2FusionWrapper
        self.encoder = Dinov2FusionWrapper(model_name="dinov2_vitl14_reg", freeze=True, encoder_feat_dim=1024)

        # [✅已转换] TransformerDecoder
        self.transformer = TransformerDecoder(
            block_type="sd3_mm_bh_cond", num_layers=transformer_layers, num_heads=transformer_heads,
            inner_dim=dim_emb, cond_dim=1536, mod_dim=None, gradient_checkpointing=True, pos_num=3660 + 1024 + 1,
        )

        # [✅已转换] MLP
        self.mlp_net = MLP(dim_emb, dim_emb, n_neurons=dim_emb, n_hidden_layers=mlp_layers, activation="silu")

        # [Jittor 修改] 必须用 nn.Parameter 才能被 load_parameters 加载
        self.id_base = nn.Parameter(jt.randn(1, 1, dim_emb))
        self.head_base = nn.Parameter(jt.randn(1, 3660, dim_emb))
        
        # [Jittor 修改] 注册 Buffer 的平替：直接赋值并停止梯度，Jittor 保存模型时会自动包含这些属性
        self.idxim = idxim.clone().stop_grad()
        self.barim = barim.clone().stop_grad()

    def execute(self, x):
        batch_size = x.shape[0]
        head_embed = self.encoder(x)

        # [Jittor 修改] F.pad -> nn.pad
        head_embed = nn.pad(head_embed, (0, 1536 - head_embed.shape[-1], 0, 0, 0, 0))

        x_b = self.head_base.repeat(batch_size, 1, 1)
        id_b = self.id_base.repeat(batch_size, 1, 1)
        
        # [Jittor 修改] torch.cat -> jt.concat
        x_in = jt.concat([x_b, id_b], dim=1)
        
        x_out = self.transformer(x_in, cond=head_embed, mod=None, temb=None)
        
        id_out = x_out[:, -1:]
        x_out = x_out[:, :-1]
        
        feature_map = self.get_uv_feature(x_out)
        feature_map = self.mlp_net(feature_map).permute(0, 3, 1, 2)
        return feature_map, id_out

    def get_uv_feature(self, feature):
        v0_map = feature[:, self.idxim[..., 0]]
        v1_map = feature[:, self.idxim[..., 1]]
        v2_map = feature[:, self.idxim[..., 2]]
        feature_map = (
            self.barim[None, ..., [0]] * v0_map
            + self.barim[None, ..., [1]] * v1_map
            + self.barim[None, ..., [2]] * v2_map
        )
        return feature_map


class Adain_Model(nn.Module):
    def __init__(self, uv_grid, dim=512):
        super(Adain_Model, self).__init__()
        # [✅已转换] GlobalGenerator_adain
        self.style_net = GlobalGenerator_adain(input_nc=dim, output_nc=12, ngf=64, style_dim=128, n_blocks=9, without_act=True)
        self.style_net.apply(weights_init)
        
        # [Jittor 修改] nn.Conv2d 保持一致，Jittor 支持
        self.to_feature = nn.Conv2d(64, 64, kernel_size=7, padding=0)
        self.mpl_to_256 = MLP(dim, 192, n_neurons=512, n_hidden_layers=2, activation="silu")
        self.uv_grid = uv_grid.clone().stop_grad()

    def execute(self, feature_map_sketch, feature_map):
        batch_size = feature_map_sketch.shape[0]
        shs_feature, shs = self.style_net.feature_forward(feature_map_sketch, feature_map)
        
        # [Jittor 修改] torch.nn.functional.grid_sample -> nn.grid_sample
        shs = nn.grid_sample(
            shs,
            self.uv_grid.repeat(batch_size, 1, 1, 1),
            align_corners=False,
            mode="bilinear",
        )[:, :, :, 0].permute(0, 2, 1)
        
        shs = _apply_color_activation(shs).reshape(batch_size, shs.shape[1], -1, 3)
        shs_feature = self.to_feature(shs_feature)

        feature_map_sketch = self.mpl_to_256(feature_map_sketch.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # [Jittor 修改] torch.cat -> jt.concat
        feature_map_out = jt.concat([feature_map_sketch, shs_feature], dim=1)
        return feature_map_out, shs


class id2w(nn.Module):
    def __init__(self, dim=512):
        super(id2w, self).__init__()
        # [✅已转换] MLP
        self.id_mlp = MLP(dim, 512 * 14, n_neurons=2048, n_hidden_layers=2, activation="silu")
        self.id_mlp2 = MLP(dim, 512 * 2, n_neurons=2048, n_hidden_layers=2, activation="silu")

    def execute(self, id_sketch, id_feat, w_predict, w_predict2, batch_size):
        w_predict = self.id_mlp(id_sketch).reshape(batch_size, 14, -1) + w_predict
        id_feat = self.id_mlp2(id_feat).reshape(batch_size, 2, -1)
        
        last_2 = w_predict[:, -2:].clone() + id_feat * w_predict[:, -2:].clone()
        w_predict[:, -2:] = last_2
        
        w_predict2[:, -1] = w_predict2[:, -1].clone() + id_feat[:, -1] * w_predict2[:, -1].clone()
        
        # [Jittor 修改] torch.cat -> jt.concat
        w_predict_out = jt.concat([w_predict, w_predict2], dim=1)
        return w_predict_out


class Feature2Gs(nn.Module):
    def __init__(self, uv_grid, flame_vertices, dim_emb=None):
        super(Feature2Gs, self).__init__()
        # [✅已转换] GSLayer
        self.gs_net = GSLayer(
            in_channels=dim_emb,
            use_rgb=False,
            sh_degree=1,
            clip_scaling=None,
            init_scaling=-10,
            init_density=0.1,
            xyz_offset=True,
            restrict_offset=True,
            xyz_offset_max_step=1.0,
            fix_opacity=False,
            fix_rotation=False,
            use_fine_feat=False,
        )
        
        self.uv_grid = uv_grid.clone().stop_grad()
        self.flame_vertices = flame_vertices.clone().stop_grad()

    def execute(self, feature_map, get_xyz=True):
        batch_size = feature_map.shape[0]
        
        # [Jittor 修改] nn.grid_sample
        x = nn.grid_sample(
            feature_map,
            self.uv_grid.repeat(batch_size, 1, 1, 1),
            align_corners=False,
            mode="bilinear",
        )[:, :, :, 0].permute(0, 2, 1)
        
        gs_attr_list = []
        gs_attr_dict = {}
        for b in range(x.shape[0]):
            gs_attr = self.gs_net(x[b], None)
            gs_attr_list.append(gs_attr)
            
        for k in gs_attr_list[0].keys():
            # [Jittor 修改] torch.stack -> jt.stack
            gs_attr_dict[k] = jt.stack([gs_attr_list[i][k] for i in range(x.shape[0])], dim=0)
            
        if get_xyz:
            gs_attr_dict["xyz"] = self.flame_vertices + gs_attr_dict["offset_xyz"]
        return gs_attr_dict


# ================================================================
# Main Class
# ================================================================

class SketchFaceGS(nn.Module):
    def __init__(self, model_cfg=None, device=None, edit_cfg=None):
        super().__init__()
        
        # [Jittor 修改] 设备管理：Jittor 全局分配显存，无需传递 device。
        if "RANK" in os.environ or (device is not None and "cuda" in str(device)):
            jt.flags.use_cuda = 1
        else:
            jt.flags.use_cuda = 1 if jt.has_cuda else 0
        self.device_str = "cuda" if jt.flags.use_cuda else "cpu"

        # 编辑超参（来自 config EDIT 段，有默认值兜底）
        self.uv_select_thresh = getattr(edit_cfg, 'uv_select_thresh', 0.1) if edit_cfg else 0.1
        self.uv_exclude_thresh = getattr(edit_cfg, 'uv_exclude_thresh', 0.9) if edit_cfg else 0.9
        self.uv_mask_dilate_size = getattr(edit_cfg, 'uv_mask_dilate_size', 20) if edit_cfg else 20

        # [✅已完成] GGHeadModel — load from exported numpy checkpoint
        # Exported by scripts/export_gghead_checkpoint.py in gs_new env
        _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        _gghead_export_dir = os.environ.get(
            "GGHEAD_EXPORT_DIR",
            os.path.join(_repo_root, "exported_gghead"),
        )
        _gghead_export_dir = os.path.abspath(_gghead_export_dir)
        _gghead_config_path = os.environ.get("GGHEAD_CONFIG_PATH")
        if _gghead_config_path is None:
            _gghead_config_candidates = [
                os.path.join(_repo_root, "assets", "gghead_config.pkl"),
                os.path.join(_gghead_export_dir, "config.pkl"),
            ]
            _gghead_config_path = next(
                (path for path in _gghead_config_candidates if os.path.exists(path)),
                _gghead_config_candidates[0],
            )
        _gghead_config_path = os.path.abspath(_gghead_config_path)

        # 1) Load config (pickle with gghead.* -> gghead_jittor.* remapping)
        class _GGHeadUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith("gghead."):
                    module = "gghead_jittor." + module[len("gghead."):]
                elif module == "gghead":
                    module = "gghead_jittor"
                return super().find_class(module, name)

        config_path = _gghead_config_path
        with open(config_path, "rb") as f:
            gghead_config = _GGHeadUnpickler(f).load()
        self.gghead = GGHeadModel(gghead_config)

        sd_path = os.environ.get("GGHEAD_STATE_DICT_PATH")
        if sd_path:
            sd_path = os.path.abspath(sd_path)
            with open(sd_path, "rb") as f:
                state_dict_np = pickle.load(f)

            param_keys = set(self.gghead.state_dict().keys())
            param_sd = {}
            buffer_sd = {}
            for k, v in state_dict_np.items():
                if k in param_keys:
                    param_sd[k] = v
                else:
                    buffer_sd[k] = v

            self.gghead.load_state_dict(param_sd)

            for k, v in buffer_sd.items():
                parts = k.split(".")
                obj = self.gghead
                try:
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], jt.array(v))
                except AttributeError:
                    pass
            del state_dict_np, param_sd, buffer_sd

        # [Jittor 修改] 冻结参数使用 stop_grad()
        for p in self.gghead.parameters():
            p.stop_grad()
        jt.gc()

        self.gghead.gghead_updata()

        # [Jittor 修改] Jittor 不需要 register_buffer，直接挂载属性并 stop_grad 即可
        self.uv_grid = self.gghead._uv_grid.clone().stop_grad()
        self.uv_idx = self.gghead._uv_idx.clone().stop_grad()
        self.idxim = self.gghead._idxim.clone().stop_grad()
        self.barim = self.gghead._barim.clone().stop_grad()
        self.faces = self.gghead._faces.clone().stop_grad()
        self.flame_vertices = self.gghead._flame_vertices.clone().stop_grad()
        
        # [✅已转换] JGaussian render() accepts jt.Var for background
        self.gaussian_bg_train = self.gghead._gaussian_bg_train.detach().clone()
        
        # [✅已转换] GaussianModel from JGaussian (Jittor)
        self._gaussian_model = GaussianModel(sh_degree=1)
        self._gaussian_model.active_sh_degree = 1
        self._gaussian_model.opacity_activation = _apply_opacity_activation

        self.random_get_sketch = random_get_sketch() # [Jittor 修改] 移除了 device 传参

        self.sketch_feature_extract = Feature_Transformer_Extract(
            dim_emb=model_cfg.EMB_DIM,
            transformer_layers=model_cfg.sketch_transformer.layers,
            transformer_heads=model_cfg.sketch_transformer.heads,
            mlp_layers=1,
            idxim=self.idxim,
            barim=self.barim,
        )
        self.color_feature_extract = Feature_Transformer_Extract(
            dim_emb=model_cfg.EMB_DIM,
            transformer_layers=model_cfg.color_transformer.layers,
            transformer_heads=model_cfg.color_transformer.heads,
            mlp_layers=1,
            idxim=self.idxim,
            barim=self.barim,
        )

        self.feature2gs = Feature2Gs(uv_grid=self.uv_grid, flame_vertices=self.flame_vertices, dim_emb=model_cfg.EMB_DIM)
        self.adain_net2 = Adain_Model(uv_grid=self.uv_grid)
        
        # [✅已转换] GFPGANv1Clean
        self.styleunet_encoder3 = GFPGANv1Clean(256, 256)
        self.id2w2 = id2w()

        self.cam_pivot = jt.array(model_cfg.cam_pivot).stop_grad()

        # Inference-only: res_conv for feature processing
        self.res_conv = ZeroConv(in_ch=256, out_ch=22, hidden_ch=512, n_layers=4)

        # [Removed for inference] loss_fn, percep_loss, net_D, criterionGAN, criterionFeat, projector

    def execute(self, batch_size=1, idx=0, sketch_img=None, device="cuda", f_image=None, change=False, gan=False, fusion=False, mask=None, conditions_gt=None, cs_in=None, color_image=None, return_baseline=False):
        # [Jittor 修改] 移除了 torch.no_grad() 上下文。Jittor 通过全局控制或外部 jt.no_grad() 包裹即可。
        if conditions_gt is not None:
            f_cs, cs, gs_model_gt, x_block, w = conditions_gt["cs"], conditions_gt["cs"], conditions_gt["gs_model"], conditions_gt["x_block"], conditions_gt["w"]
            color_image = conditions_gt['color_image']
            t_image = f_image = self.gs_gen(gs_model=conditions_gt["gs_model"], c=cs_in)
            if cs_in is not None:
                f_cs = cs = cs_in
        else:
            if f_image is not None:
                t_image = f_image
                if cs_in is not None:
                    cs = cs_in
                else:
                    cs = rand_c2w(self.cam_pivot, batch_size)
                f_cs = cs
            else:
                f_image, t_image, f_cs, cs, gs_model_gt, x_block, w = self.gen_image(
                    seed=idx,
                    batch_size=batch_size,
                    cs_in=cs_in,
                    noise_mode="none",
                    return_xblock=True,
                )

        if return_baseline:
            baseline_conditions = {
                "x_block": locals().get("x_block"),
                "gs_model": locals().get("gs_model_gt"),
                "cs": locals().get("cs"),
                "w": locals().get("w"),
                "color_image": color_image if color_image is not None else f_image,
            }
            return {
                "f_image": f_image,
                "t_image": t_image,
                "gen_image": f_image,
                "feedforward_image": None,
                "adain_image": None,
                "fusion_image": None,
                "conditions": baseline_conditions,
                "w_predict": locals().get("w"),
                "x_block_gen": locals().get("x_block"),
            }

        if change:
            idx = random.randint(0, 200000)
            f_image_sketch, t_image, _, _, _ = self.gen_image(seed=idx, batch_size=batch_size)
            sketch = self.random_get_sketch(f_image_sketch, a=None)
        else:
            sketch = self.random_get_sketch(f_image, a=None) if sketch_img is None else sketch_img

        color_feature_map, color_id = self.color_feature_extract(color_image if color_image is not None else f_image)
        sketch_feature_map, sketch_id = self.sketch_feature_extract(sketch)

        color_gs = self.feature2gs(color_feature_map)
        sketch_gs = self.feature2gs(sketch_feature_map)
        sketch_gs["shs"] = color_gs["shs"]
        feedforward_gs = sketch_gs

        feature_map, shs = self.adain_net2(sketch_feature_map, color_feature_map)

        w_predict, conditions, gs_att_out, w_predict2 = self.styleunet_encoder3(feature_map, return_style2=True)
        w_predict = self.id2w2(sketch_id, color_id, w_predict, w_predict2, batch_size)

        res_plane = None

        feedforward_image = self.gs_gen(feedforward_gs, c=cs)

        adain_gs = feedforward_gs
        # [Jittor 修改] 移除 .contiguous()，Jittor 内部会自动处理连续性
        adain_gs["shs"] = shs 
        adain_image = self.gs_gen(adain_gs, c=cs)

        gen_image, gs_model, x_block_gen = self.gen_image(
            batch_size=batch_size, w_in=w_predict, cs_in=cs, conditions=conditions, prepare_data=False, res_plane=res_plane, return_xblock=True,
        )
        gen_image = self.gs_gen(gs_model=gs_model, c=cs)

        if fusion and conditions_gt is None:
            # No external fusion cache yet: use current forward outputs as bootstrap anchors.
            x_block = x_block_gen
            w = w_predict
            gs_model_gt = gs_model
            f_cs = cs

        fusion_image = None
        if fusion:
            uv_mask = self.mask2uv_mask(mask, batch_size, c2cam(f_cs), gs_model_gt, gs_model)
            conditions_f = {"conditions": conditions, "mask_gt": uv_mask, "x_block": x_block, "w": w, "mask_gt_last": uv_mask}
            fusion_image, gs_model, x_block = self.gen_image(
                batch_size=batch_size, w_in=w_predict, cs_in=cs, conditions=conditions_f, prepare_data=False, res_plane=res_plane, return_xblock=True,
            )
            fusion_image = self.gs_gen(gs_model=gs_model, c=cs)

        conditions_fornext = {
            "x_block": locals().get("x_block"), "gs_model": locals().get("gs_model_gt", gs_model), "cs": cs,
            "w": locals().get("w"),
            "color_image": color_image if color_image is not None else f_image,
        }
        
        return {
            "f_image": f_image, "t_image": t_image, "gen_image": gen_image,
            "feedforward_image": feedforward_image, "adain_image": adain_image,
            "fusion_image": fusion_image,
            "conditions": conditions_fornext, "w_predict": w_predict, "x_block_gen": x_block_gen,
        }

    def gen_image(self, seed=0, device="cuda:0", w_in=None, cs_in=None, batch_size=1, truncation_psi=0.7, resolution=512, conditions=None, prepare_data=True, res_plane=None, return_xblock=False, return_conditions=False, noise_mode="const"):
        if cs_in is not None:
            cs = cs_in
        else:
            cs = rand_c2w(self.cam_pivot, batch_size * 2)

        c_front = encode_camera_params(
            Pose(
                matrix_or_rotation=np.eye(3),
                translation=(0, 0.0, 3.5),
                pose_type=PoseType.CAM_2_WORLD,
                camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL,
            ),
            DEFAULT_INTRINSICS,
        )
        
        # [Jittor 修改] torch.from_numpy -> jt.array(). 忽略 .to(device)
        c_front = jt.array(c_front).unsqueeze(0).repeat(batch_size, 1)
        sh_ref_cam, intrinsics = decode_camera_params(c_front[0].numpy()) # .cpu() 替换为 .numpy()

        if w_in is not None:
            w = w_in
        else:
            # [Jittor 修改] 随机数生成：直接使用 jt.randn，设置全局 seed 可复现
            jt.set_global_seed(seed)
            z = jt.randn((batch_size, self.gghead._config.z_dim))
            w = self.gghead.mapping(z, c_front, truncation_psi=truncation_psi)
            w = w.repeat(cs.shape[0] // w.shape[0], 1, 1)

        if return_xblock or return_conditions:
            x_block, output = self.gghead.synthesis(
                w, cs, sh_ref_cam=sh_ref_cam, noise_mode=noise_mode, neural_rendering_resolution=resolution, conditions=conditions, return_xblock=True, planes_res=res_plane,
            )
            for n, x in enumerate(x_block):
                x_block[n] = x[:batch_size]
            if return_conditions:
                conditions = {
                    "x_block": x_block, "gs_model": clone_gaussian_model(self.gghead._gaussian_model),
                    "cs": cs, "w": w, "color_image": (output["image"][:batch_size] + 1) / 2,
                }
                return conditions
        else:
            output = self.gghead.synthesis(
                w, cs, sh_ref_cam=sh_ref_cam, noise_mode=noise_mode, neural_rendering_resolution=resolution, conditions=conditions, return_xblock=False, planes_res=res_plane,
            )
            
        if prepare_data:
            f_images = (output["image"][:batch_size] + 1) / 2
            t_images = (output["image"][batch_size:] + 1) / 2
            return (f_images, t_images, cs[:batch_size], cs[batch_size:], clone_gaussian_model(self.gghead._gaussian_model), x_block if return_xblock else None, w[:batch_size])
        else:
            images = (output["image"][:batch_size] + 1) / 2
            return (images, clone_gaussian_model(self.gghead._gaussian_model), x_block if return_xblock else None)

    # [Removed for inference] calc_metrics — training losses removed

    def gs_gen(self, gs_attr_dict=None, c=None, bg=None, device=None, gs_model=None):
        # [✅已转换] Pure Jittor — no torch bridge needed
        cam = c2cam(c)
        gen_images = []

        if not hasattr(self, '_cached_gaussian_sh_ref_cam'):
            c_front = encode_camera_params(
                Pose(
                    matrix_or_rotation=np.eye(3),
                    translation=(0, 0.0, 3.5),
                    pose_type=PoseType.CAM_2_WORLD,
                    camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL,
                ),
                DEFAULT_INTRINSICS,
            )
            c_front = jt.array(c_front).unsqueeze(0)
            sh_ref_cam, intrinsics = decode_camera_params(c_front[0].numpy())
            intrinsics = intrinsics.rescale(512, inplace=False)
            self._cached_gaussian_sh_ref_cam = pose_to_rendercam(sh_ref_cam, intrinsics, 512, 512)
        gaussian_sh_ref_cam = self._cached_gaussian_sh_ref_cam

        sh_degree = self.gghead._config.gaussian_attribute_config.sh_degree
        n_feature_channels = self.gghead._config.gaussian_attribute_config.n_color_channels

        if gs_attr_dict:
            bsize = gs_attr_dict["xyz"].shape[0]
            for i in range(bsize):
                # JGaussian GaussianModel uses jt.Var directly
                self._gaussian_model._xyz = gs_attr_dict["xyz"][i]
                self._gaussian_model._scaling = gs_attr_dict["scaling"][i]
                self._gaussian_model._rotation = gs_attr_dict["rotation"][i]
                self._gaussian_model._opacity = gs_attr_dict["opacity"][i]
                self._gaussian_model._features_dc = gs_attr_dict["shs"][i][:, [0]]
                self._gaussian_model._features_rest = gs_attr_dict["shs"][i][:, 1:]
                self._gaussian_model.screenspace_points = jt.zeros((gs_attr_dict["xyz"][i].shape[0], 3))

                shs_view = self._gaussian_model.get_features.view(
                    -1, (sh_degree + 1) ** 2, n_feature_channels).permute(0, 2, 1)
                dir_pp = (self._gaussian_model.get_xyz
                          - gaussian_sh_ref_cam.camera_center.repeat(1, 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
                sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
                colors = jt.clamp(sh2rgb + 0.5, min_v=0.0)
                override_color = colors

                bg_input = bg if bg is not None else self.gaussian_bg_train
                rendered = render(cam[i], self._gaussian_model, PipelineParams2(),
                                  bg_input, override_color=override_color)
                rendered_image = (rendered["render"] if isinstance(rendered, dict) and "render" in rendered
                                  else rendered)
                gen_images.append(rendered_image)

        elif gs_model:
            gs_model.screenspace_points = jt.zeros((gs_model.get_xyz.shape[0], 3))
            shs_view = gs_model.get_features.view(
                -1, (sh_degree + 1) ** 2, n_feature_channels).permute(0, 2, 1)
            dir_pp = gs_model.get_xyz - gaussian_sh_ref_cam.camera_center.repeat(1, 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
            sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
            colors = jt.clamp(sh2rgb + 0.5, min_v=0.0)
            override_color = colors

            bg_input = bg if bg is not None else self.gaussian_bg_train
            for i in range(len(cam)):
                rendered = render(cam[i], gs_model, PipelineParams2(),
                                  bg_input, override_color=override_color)
                rendered_image = (rendered["render"] if isinstance(rendered, dict) and "render" in rendered
                                  else rendered)
                gen_images.append(rendered_image)

        gen_images = jt.stack(gen_images, dim=0)
        return gen_images

    def get_mask_index(self, gaussian_model, mask, cam, value=0.7, return_weights=False):
        # [✅已转换] JGaussian apply_weights returns (weights, weights_cnt) directly
        if not isinstance(mask, jt.Var):
            mask = jt.array(mask)
        weights, weights_cnt = gaussian_model.apply_weights(cam, mask)
        weights_return = weights.clone()
        weights = weights / (weights_cnt + 1e-7)
        selected_mask = weights > value
        selected_mask = selected_mask[:, 0]
        if return_weights:
            return selected_mask, weights_return
        return selected_mask

    def get_mask_uv(self, gs, batch_size, cam, mask=None, value1=None, value2=None):
        if value1 is None:
            value1 = self.uv_select_thresh
        if value2 is None:
            value2 = self.uv_exclude_thresh
        uv_mask_ = []
        uv_mask_more_ = []
        for i in range(batch_size):
            if mask is None:
                mask_, cam_ = random_irregular_mask_smooth(512, 512, area=[100, 2]), cam[i]
            else:
                mask_, cam_ = mask[i], cam[i]
            
            mask_index = self.get_mask_index(gs, mask_, cam_, value=value1)
            # mask_index is already jt.Var (bool)
            coords = self.gghead._uv_idx[mask_index]

            uv_mask = jt.zeros((512, 512))
            uv_mask[coords[:, 1], coords[:, 0]] = 1.0
            uv_mask_.append(uv_mask)

            mask_gt = 1.0 - mask_
            mask_index_gt = self.get_mask_index(gs, mask_gt, cam_, value=value2)
            mask_index_gt_inv = jt.logical_not(mask_index_gt)
            coords_gt = self.gghead._uv_idx[mask_index_gt_inv]

            uv_mask_more = jt.zeros((512, 512))
            uv_mask_more[coords_gt[:, 1], coords_gt[:, 0]] = 1.0
            uv_mask_more_.append(uv_mask_more)

        return (
            (jt.stack(uv_mask_, dim=0)[:, None, ...]).bool(),
            (jt.stack(uv_mask_more_, dim=0)[:, None, ...]).bool(),
        )

    def mask2uv_mask(self, mask, batch_size, f_cam, gs_gt, gs):
        gt_uv_mask, gt_uv_mask_more = self.get_mask_uv(gs_gt, batch_size, f_cam, mask=mask)
        gen_uv_mask, gen_uv_mask_more = self.get_mask_uv(gs, batch_size, f_cam, mask=mask)

        uv_mask_less = gt_uv_mask | gen_uv_mask
        uv_mask_more = gt_uv_mask_more | gen_uv_mask_more

        uv_mask_less = dilate_mask_scipy(uv_mask_less, self.uv_mask_dilate_size)
        uv_mask = uv_mask_less & uv_mask_more
        return uv_mask

    # [Removed for inference] run_inversion_our — depends on Projector (training)

    def get_video(self, c):
        cs_list = get_cs_list(c)
        pictures = self.gs_gen(gs_attr_dict=None, c=cs_list, bg=None, gs_model=self.gghead._gaussian_model)

        # Jittor tensor (C,H,W) 0-1 → numpy (H,W,C) uint8
        frames = []
        for pic in pictures:
            arr = np.clip(pic.numpy().transpose(1, 2, 0), 0, 1)
            frames.append((arr[..., :3] * 255).astype(np.uint8))

        output_folder = "output/sampled_heads/"
        os.makedirs(output_folder, exist_ok=True)

        import mediapy
        seed = random.randint(0, 10000)
        mediapy.write_video(f"{output_folder}/{seed:04d}.mp4", frames, fps=16)