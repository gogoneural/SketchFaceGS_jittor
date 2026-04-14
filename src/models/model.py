from operator import ge
import os
from re import S
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import random
import cv2

# -----------------------------------------------------------------------------

# Core utilities (dreifus / eg3d)
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix import Pose
from eg3d.datamanager.nersemble import encode_camera_params, decode_camera_params

# GGHead
from gghead.constants import DEFAULT_INTRINSICS
from gghead.model_manager.finder import find_model_manager
from gghead.models.gghead_model import GGHeadModel

# LHM (local third_party)
from LHM.models.encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper
from LHM.models.transformer import TransformerDecoder
from LHM.models.rendering.gs_renderer import GSLayer
from LHM.models.rendering.utils.utils import MLP

# Gaussian Splatting
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.arguments import PipelineParams2
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.sh_utils import C0, eval_sh
from gaussian_splatting.scene.cameras import pose_to_rendercam

# Project internal modules
from src.models.modules.gfpganv1_clean_arch import GFPGANv1Clean
from src.models.modules.networks_maskGAN import weights_init, GlobalGenerator_adain
from src.models.modules.conv import ZeroConv
from src.models.libs.flame_model import FLAMEModel

from src.utils.camera_utils import c2cam, rand_c2w, FaceRecon
from src.utils.gaussian_utils import clone_gaussian_model
from src.utils.perceptual_utils import FacePerceptualLoss
from src.utils.sketch_utils import random_get_sketch
from src.utils.video_utils import get_cs_list
from ..utils.gaussian_utils import _apply_opacity_activation, _apply_color_activation
from pix2pixHD.model.networks import define_D, GANLoss

# -----------------------------------------------------------------------------
# 5. Global Configuration & Initialization
# -----------------------------------------------------------------------------
# # Constants
# OFFSET_MAX = 0.2
# SCALE_MAX = 0.02
# EMB_DIM = 512
# F_SIZE = 512

# Data Transforms
transform = transforms.Compose(
    [
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False),
    ]
)

# Initialize FLAME Model
# 注意：通常建议将模型初始化放在 main 函数或类的 __init__ 中，
# 这里保留全局初始化以匹配原代码逻辑。
print("Initializing FLAME model...")
flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=5, no_lmks=True)
faces = flame_model.get_faces()
i0 = faces[..., 0]
i1 = faces[..., 1]
i2 = faces[..., 2]
# cam_pivot = torch.tensor([0, 0.05, 0.2])


# ================================================================
# Helper Classes (dependencies for SketchFaceGS)
# ================================================================


def random_irregular_mask_smooth(
    height: int,
    width: int,
    min_area: int = None,
    max_area: int = None,
    connectivity: int = 4,
    seed: int = None,
    blur_ksize: int = 15,
    blur_sigma: float = 5.0,
    thr: float = 0.5,
    area: list = [20, 3],
) -> torch.Tensor:
    """
    生成随机连通掩码并做高斯模糊+阈值圆滑处理，返回 shape=(1,H,W) 的 float32 tensor。

    Args:
        height, width: 掩码高宽。
        min_area, max_area: 连通区域像素数范围（默认 H*W//20 ~ H*W//2）。
        connectivity: 连通性（4 或 8）。
        seed: 随机种子。
        blur_ksize: 高斯核大小（必须为奇数）&#8203;:contentReference[oaicite:3]{index=3}。
        blur_sigma: 高斯标准差&#8203;:contentReference[oaicite:4]{index=4}。
        thr: 重阈值（0–1 之间），默认 0.5&#8203;:contentReference[oaicite:5]{index=5}。

    Returns:
        torch.Tensor: shape=(1, H, W)，值为 0.0 或 1.0 的圆滑二值掩码。
    """
    # ——区域生长生成不规则连通区域（BFS 种子生长）——
    if seed is not None:
        np.random.seed(
            seed
        )  # 可复现种子选择&#8203;:contentReference[oaicite:6]{index=6}

    total = height * width
    if min_area is None:
        min_area = total // area[0]
    if max_area is None:
        max_area = total // area[1]
    # 随机目标区域大小&#8203;:contentReference[oaicite:7]{index=7}
    target_size = np.random.randint(min_area, max_area + 1)

    # 随机选取种子点
    start = (np.random.randint(height), np.random.randint(width))
    region = {start}
    frontier = [start]

    # 邻域定义
    if connectivity == 8:
        neighs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        neighs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 随机游走式扩展以获得不规则边界&#8203;:contentReference[oaicite:8]{index=8}
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

    # 构造原始生长掩码
    mask0 = np.zeros((height, width), dtype=np.float32)
    for r, c in region:
        mask0[r, c] = 1.0

    # ——高斯模糊 + 重阈值圆滑边缘——
    # OpenCV GaussianBlur 要求输入 uint8 或 float32，范围 0–255 或 0–1 均可
    mask_uint8 = (mask0 * 255).astype(
        np.uint8
    )  # 转为 0–255 格式&#8203;:contentReference[oaicite:9]{index=9}
    # 高斯模糊&#8203;:contentReference[oaicite:10]{index=10}
    blur = cv2.GaussianBlur(mask_uint8, (blur_ksize, blur_ksize), blur_sigma)
    # 重阈值回二值（127 对应 0.5）&#8203;:contentReference[oaicite:11]{index=11}
    _, mask_thr = cv2.threshold(blur, int(thr * 255), 255, cv2.THRESH_BINARY)
    # 转回 0.0/1.0 float32
    mask_smooth = mask_thr.astype(np.float32) / 255.0

    # 返回单通道 tensor (1, H, W)&#8203;:contentReference[oaicite:12]{index=12}
    return torch.from_numpy(mask_smooth).unsqueeze(0)


class Feature_Transformer_Extract(nn.Module):
    def __init__(
        self,
        dim_emb=512,
        transformer_layers=4,
        transformer_heads=16,
        mlp_layers=1,
        idxim=None,
        barim=None,
    ):
        super(Feature_Transformer_Extract, self).__init__()
        self.encoder = Dinov2FusionWrapper(
            model_name="dinov2_vitl14_reg",
            freeze=True,
            encoder_feat_dim=1024,
        )
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        # ------------------------------------ transformer ------------------------------------------------------------
        self.transformer = TransformerDecoder(
            block_type="sd3_mm_bh_cond",
            num_layers=transformer_layers,  # 15
            num_heads=transformer_heads,
            inner_dim=dim_emb,
            cond_dim=1536,
            mod_dim=None,
            gradient_checkpointing=True,
            pos_num=3660 + 1024 + 1,
        )

        # ------------------------------------------- mlp behind transformer -------------------------------------------
        self.mlp_net = MLP(
            dim_emb,
            dim_emb,
            n_neurons=dim_emb,
            n_hidden_layers=mlp_layers,
            activation="silu",  # 1
        )

        self.id_base = nn.Parameter(
            torch.randn(1, 1, dim_emb), requires_grad=True
        )  # 1024
        self.head_base = nn.Parameter(
            torch.randn(1, 3660, dim_emb), requires_grad=True
        )  # 3660
        self.register_buffer("idxim", idxim.clone())
        self.register_buffer("barim", barim.clone())

    def forward(self, x):
        batch_size = x.shape[0]
        ################################################################################
        # with torch.cuda.amp.autocast(enabled=False):

        head_embed = self.encoder(x)  # backbone 以 float32 正常运行

        head_embed = F.pad(
            head_embed, (0, 1536 - head_embed.shape[-1], 0, 0, 0, 0)
        )  # the same as sd3, learnable

        x = self.head_base.repeat(batch_size, 1, 1)
        id = self.id_base.repeat(batch_size, 1, 1)
        x = torch.cat([x, id], 1)
        x = self.transformer(
            x,
            cond=head_embed,
            mod=None,
            temb=None,
        )  # [B, L
        #
        id = x[:, [-1]]  # Bxd
        x = x[:, :-1]
        feature_map = self.get_uv_feature(x)
        feature_map = self.mlp_net(feature_map).permute(0, 3, 1, 2)
        return feature_map, id

    def get_uv_feature(self, feature):
        v0_map = feature[:, self.idxim[..., 0]]
        v1_map = feature[
            :,
            self.idxim[
                ...,
                1,
            ],
        ]
        v2_map = feature[:, self.idxim[..., 2]]
        feature_map = (
            self.barim[None, ..., [0]] * v0_map
            + self.barim[None, ..., [1]] * v1_map
            + self.barim[None, ..., [2]] * v2_map
        )
        return feature_map


from scipy.ndimage import binary_dilation


def dilate_mask_scipy(mask: torch.Tensor, n: int) -> torch.Tensor:
    """
    使用 SciPy 对布尔掩码进行膨胀操作。

    Args:
        mask (torch.Tensor): 输入的布尔掩码，形状为 (1, 1, H, W)。
        n (int): 膨胀窗口的边长。

    Returns:
        torch.Tensor: 膨胀后的布尔掩码，形状与输入相同。
    """
    # 1. 将 PyTorch Tensor 转换为 NumPy 数组
    #    .squeeze() 移除前两个维度 (1, 1)，因为 scipy 函数处理 2D 数组
    mask_numpy = mask.squeeze().cpu().numpy()

    # 2. 定义膨胀的结构元素（一个 n x n 的全为 True 的矩阵）
    structure = np.ones((n, n), dtype=bool)

    # 3. 执行二值膨胀
    dilated_mask_numpy = binary_dilation(mask_numpy, structure=structure)

    # 4. 将结果转换回 PyTorch Tensor
    #    先转换为 Tensor，然后 .unsqueeze(0).unsqueeze(0) 恢复 (1, 1, H, W) 的形状
    dilated_mask = (
        torch.from_numpy(dilated_mask_numpy).unsqueeze(0).unsqueeze(0).to(mask.device)
    )

    return dilated_mask


class Adain_Model(nn.Module):
    def __init__(self, uv_grid, dim=512):
        super(Adain_Model, self).__init__()
        self.style_net = GlobalGenerator_adain(
            input_nc=dim,
            output_nc=12,
            ngf=64,
            style_dim=128,
            n_blocks=9,
            without_act=True,
        )
        self.style_net.apply(weights_init)
        self.to_feature = nn.Conv2d(64, 64, kernel_size=7, padding=0)
        self.mpl_to_256 = MLP(
            dim, 192, n_neurons=512, n_hidden_layers=2, activation="silu"
        )
        self.register_buffer("uv_grid", uv_grid.detach().clone())

    def forward(self, feature_map_sketch, feature_map):
        batch_size = feature_map_sketch.shape[0]
        shs_feature, shs = self.style_net.feature_forward(
            feature_map_sketch, feature_map
        )
        shs = torch.nn.functional.grid_sample(
            shs,
            self.uv_grid.repeat(batch_size, 1, 1, 1),
            align_corners=False,
            mode="bilinear",
        )[:, :, :, 0].permute(
            0, 2, 1
        )  # [G, D]
        shs = _apply_color_activation(shs).reshape(batch_size, shs.shape[1], -1, 3)

        # shs_feature = self.shs_encoder(shs.permute(0, 2, 3, 1))
        # shs_feature = self.mlp_to64(shs_feature).permute(0, 3, 1, 2)

        shs_feature = self.to_feature(shs_feature)

        feature_map_sketch = self.mpl_to_256(
            feature_map_sketch.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)

        feature_map = torch.cat([feature_map_sketch, shs_feature], 1)
        return feature_map, shs  # 256


class id2w(nn.Module):
    def __init__(
        self,
        dim=512,
    ):
        super(id2w, self).__init__()
        self.id_mlp = MLP(
            dim, 512 * 14, n_neurons=2048, n_hidden_layers=2, activation="silu"
        )
        self.id_mlp2 = MLP(
            dim, 512 * 2, n_neurons=2048, n_hidden_layers=2, activation="silu"
        )

    def forward(self, id_sketch, id, w_predict, w_predict2, batch_size):
        w_predict = self.id_mlp(id_sketch).reshape(batch_size, 14, -1) + w_predict
        id = self.id_mlp2(id).reshape(batch_size, 2, -1)
        last_2 = w_predict[:, -2:].clone() + id * w_predict[:, -2:].clone()
        w_predict[:, -2:] = last_2
        # -------------------------去了-------------------------------------------
        w_predict2[:, -1] = (
            w_predict2[:, -1].clone() + id[:, -1] * w_predict2[:, -1].clone()
        )
        w_predict = torch.cat([w_predict, w_predict2], 1)
        return w_predict


class Feature2Gs(nn.Module):
    def __init__(
        self,
        uv_grid,
        flame_vertices,
        dim_emb=None,
    ):
        super(Feature2Gs, self).__init__()
        self.gs_net = GSLayer(
            in_channels=dim_emb,
            use_rgb=False,
            sh_degree=1,
            clip_scaling=None,  # 0.2
            init_scaling=-10,
            init_density=0.1,
            xyz_offset=True,
            restrict_offset=True,
            xyz_offset_max_step=1.0,
            fix_opacity=False,
            fix_rotation=False,
            use_fine_feat=(
                # True
                # if decode_with_extra_info is not None
                #    and decode_with_extra_info["type"] is not None
                False
            ),
        )
        # register uv grid as buffer so it follows model.to(device)
        self.register_buffer("uv_grid", uv_grid.detach().clone())
        self.register_buffer("flame_vertices", flame_vertices.detach().clone())

    def forward(self, feature_map, get_xyz=True):
        batch_size = feature_map.shape[0]
        x = torch.nn.functional.grid_sample(
            feature_map,
            self.uv_grid.repeat(batch_size, 1, 1, 1),
            align_corners=False,
            mode="bilinear",
        )[:, :, :, 0].permute(
            0, 2, 1
        )  # [G, D]
        gs_attr_list = []
        gs_attr_dict = {}
        for b in range(x.shape[0]):
            gs_attr = self.gs_net(
                x[b],
                None,
            )
            gs_attr_list.append(gs_attr)
        for k in gs_attr_list[0].keys():
            gs_attr_dict[k] = torch.stack(
                [gs_attr_list[i][k] for i in range(x.shape[0])], 0
            )
        if get_xyz:
            gs_attr_dict["xyz"] = self.flame_vertices + gs_attr_dict["offset_xyz"]
        return gs_attr_dict


# ================================================================
# Main Class (depends on helper classes)
# ================================================================


class SketchFaceGS(nn.Module):
    def __init__(
        self,
        model_cfg=None,
        device=None
    ):
        super().__init__()
        # ------------------------------------- device --------------------------------------------------------
        if "RANK" in os.environ:  # 多卡环境
            rank = int(os.environ["RANK"])
            self.device = torch.device(f"cuda:{rank}")
        else:  # 单卡环境
            self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        # ------------------------------------ gghead initialization --------------------------------------------------------
        model_manager = find_model_manager("GGHEAD-1_ffhq512")
        checkpoint = model_manager._resolve_checkpoint_id(-1)
        gghead = model_manager.load_checkpoint(
            checkpoint, load_ema=True
        )  # .to(device)self._config
        self.gghead = GGHeadModel(gghead._config)
        state_dict = gghead.state_dict()
        self.gghead.load_state_dict(state_dict, strict=False)
        for p in self.gghead.parameters():
            p.requires_grad = False
        # 再单独 unfreeze name 中含有 'torgb_sketch' 的那些参数
        del gghead
        # 【推荐步骤】清空未使用的CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ----------------------------------------------------------------------
        self.gghead.gghead_updata()
        # Register tensors as buffers so they move with `model.to(device)` and
        # are saved/loaded with the module state dict.
        self.register_buffer("uv_grid", self.gghead._uv_grid.detach().clone())
        self.register_buffer("uv_idx", self.gghead._uv_idx.detach().clone())
        self.register_buffer("idxim", self.gghead._idxim.detach().clone())
        self.register_buffer("barim", self.gghead._barim.detach().clone())
        self.register_buffer("faces", self.gghead._faces.detach().clone())
        self.register_buffer(
            "flame_vertices", self.gghead._flame_vertices.detach().clone()
        )
        self.gghead._uv_idx = self.gghead._uv_idx.to(self.device)
        self.gghead.to(self.device)
        # -------------------------------- gaussian model initialization --------------------------------------------------------
        # register background as buffer (will be moved by model.to(device))
        self.register_buffer(
            "gaussian_bg_train", self.gghead._gaussian_bg_train.detach().clone()
        )
        self._gaussian_model = GaussianModel(sh_degree=1)
        self._gaussian_model.active_sh_degree = 1
        self._gaussian_model.opacity_activation = _apply_opacity_activation

        # ------------------------------------ gen sketch ------------------------------------------------------------
        self.random_get_sketch = random_get_sketch(self.device)

        # --------------------------------- feature_exract --------------------------------------------------
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
        # ------------------------------------ feature2gs --------------------------------------------------------
        self.feature2gs = Feature2Gs(
            uv_grid=self.uv_grid,
            dim_emb=model_cfg.EMB_DIM,
            flame_vertices=self.flame_vertices,
        )
        # ------------------------------------ adain_net --------------------------------------------------------
        self.adain_net2 = Adain_Model(uv_grid=self.uv_grid)

        # ------------------------------------ styleunet encoder --------------------------------------------------------
        self.styleunet_encoder3 = GFPGANv1Clean(256, 256)

        # ------------------------------------ id2w --------------------------------------------------------
        self.id2w2 = id2w()

        # register cam pivot from config or default

        self.register_buffer("cam_pivot", torch.tensor(model_cfg.cam_pivot))

        # ----------------- loss ----------------------------------
        self.loss_fn = nn.functional.l1_loss
        self.percep_loss = FacePerceptualLoss(loss_type="l1", weighted=True)

        # ----------------------------- face cam get --------------------------------
        self.face_recon_model = FaceRecon(self.device)

        # -------------------------------res mlp --------------------------------
        self.res_conv = ZeroConv(
            in_ch=256,
            out_ch=22,
            hidden_ch=512,
            n_layers=4,
        )

        # ---------------------------------- gan ---------------------------------------------
        self.net_D = define_D(
            input_nc=3,
            ndf=64,
            n_layers_D=3,
            use_sigmoid=False,
            num_D=3,
            getIntermFeat=True,
        )
        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        self.criterionFeat = torch.nn.L1Loss()
        
        #--------------------------------- inversion ---------------------------------------------
        from src.utils.inversion import Projector
        
        self.projector = Projector(
            G=self.gghead,
            device=self.device,
            num_iter=150,
            log_images_step=10,
            percep_lambda=1.0,
            lr=3e-2,
            face_recon_model=self.face_recon_model,
        )
    def forward(
        self,
        batch_size=1,
        idx=0,
        sketch_img=None,
        device="cuda",
        f_image=None,
        change=False,
        gan=False,
        fusion=False,
        mask=None,
        conditions_gt=None,
        cs_in = None,
        color_image = None,
    ):
        # 设备优先从 self._get_device() 获取，避免分散使用不同来源的 device
        # time1 = time.time()

        dev = self.device
        # with torch.no_grad():
        # ----------------------------- input --------------------------------------------------------
        if conditions_gt is not None:

            f_cs, cs, gs_model_gt, x_block, w = (
                conditions_gt["cs"],
                conditions_gt["cs"],
                conditions_gt["gs_model"],
                conditions_gt["x_block"],
                conditions_gt["w"],
            )
            color_image = conditions_gt['color_image']
            t_image = f_image = self.gs_gen(
                gs_model=conditions_gt["gs_model"],
                c=cs_in,
                device=self.device,
            )
            if cs_in!=None:
                f_cs = cs = cs_in
        else:
            if f_image != None:
                t_image = f_image
                # f_image = f_image.float()
                # sketch_img = sketch_img.float()

                _, cs = self.face_recon_model((f_image + 1) / 2)
                # cs = c_3[0]

            else:

                f_image, t_image, f_cs, cs, gs_model_gt, x_block, w = self.gen_image(
                    seed=idx,
                    batch_size=batch_size,
                    device=self.device,
                    return_xblock=fusion,
                )
            # time2 = time.time()
        if change:
            idx = random.randint(0, 200000)
            f_image_sketch, t_image, _, _, _ = self.gen_image(
                seed=idx, batch_size=batch_size, device=self.device
            )
            sketch = self.random_get_sketch(f_image_sketch, a=None)
        else:
            sketch = (
                self.random_get_sketch(f_image, a=None)
                if sketch_img == None
                else sketch_img
            )

        # ---------------------------- feature extact --------------------------------------------------------
        # with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        color_feature_map, color_id = self.color_feature_extract(color_image if color_image is not None else f_image)
        sketch_feature_map, sketch_id = self.sketch_feature_extract(sketch)
        # time3 = time.time()

        color_gs = self.feature2gs(color_feature_map)
        sketch_gs = self.feature2gs(sketch_feature_map)
        sketch_gs["shs"] = color_gs["shs"]
        feedforward_gs = sketch_gs
        # --------------------------------- adain ------------------------------------------------------------------
        feature_map, shs = self.adain_net2(sketch_feature_map, color_feature_map)

        # ---------------------------- styleunet encoder --------------------------------------------------------
        w_predict, conditions, gs_att_out, w_predict2 = self.styleunet_encoder3(
            feature_map, return_style2=True
        )
        w_predict = self.id2w2(sketch_id, color_id, w_predict, w_predict2, batch_size)
        # time4 = time.time()
        ########################################################################################

        res_plane = None  # self.res_conv(feature_map)

        feedforward_image = self.gs_gen(feedforward_gs, c=cs, device=self.device)

        adain_gs = feedforward_gs
        adain_gs["shs"] = shs.contiguous()
        adain_image = self.gs_gen(adain_gs, c=cs, device=self.device)
        ########################################################################################

        # --------------------------------- gen image ------------------------------------------------------------------
        #
        #
        #
        #
        #
        gen_image, gs_model, x_block_gen = self.gen_image(
            device=dev,
            batch_size=batch_size,
            w_in=w_predict,
            cs_in=cs,
            conditions=conditions,
            prepare_data=False,
            res_plane=res_plane,
            return_xblock=True,
        )
        gen_image = self.gs_gen(gs_model=gs_model, c=cs, device=self.device)
        # --------------------------------fusion---------------------------------------------------------------------------
        if fusion:

            uv_mask = self.mask2uv_mask(
                mask, batch_size, c2cam(f_cs), gs_model_gt, gs_model
            )
            
            conditions = {
                "conditions": conditions,
                "mask_gt": uv_mask,
                "x_block": x_block,
                "w": w,
                "mask_gt_last": uv_mask,
            }
            fusion_image, gs_model, x_block = self.gen_image(
                device=dev,
                batch_size=batch_size,
                w_in=w_predict,
                cs_in=cs,
                conditions=conditions,
                prepare_data=False,
                res_plane=res_plane,
                return_xblock=True,
            )
            fusion_image = self.gs_gen(gs_model=gs_model, c=cs, device=self.device)
            
            # mask_index = self.get_mask_index(
            #     gs_model, mask[0], c2cam(cs)[0], value=0.1
            # )
            # gs_model._features_dc[mask_index] = torch.full_like(gs_model._features_dc[mask_index], 0.5)
            
            # self.gs_gen(gs_model=gs_model, c=cs, device=self.device)

        # time5 = time.time()
        if gan:
            pred_fake_pool = self.net_D(gen_image.detach())
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)

            # Real Detection and Loss
            pred_real = self.net_D(t_image.detach())
            loss_D_real = self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)
            pred_fake = self.net_D.forward(gen_image)
            loss_G_GAN = self.criterionGAN(pred_fake, True)

            loss_G_GAN_Feat = 0

            for i in range(3):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach()
                    )
            gan_loss = {"loss_G_GAN": loss_G_GAN, "loss_G_GAN_Feat": loss_G_GAN_Feat}
            D_loss = {
                "loss_D_fake": loss_D_fake,
                "loss_D_real": loss_D_real,
            }
        
        
        
        conditions_fornext = {
            "x_block": locals().get("x_block"),
            "gs_model": gs_model,
            "cs": cs,
            "w": locals().get("w"),
            "conditions_gt": conditions,
            "color_image": color_image if color_image is not None else f_image,
        }
        results = {
            "f_image": f_image,
            "t_image": t_image,
            "gen_image": gen_image,
            "feedforward_image": feedforward_image,
            "adain_image": adain_image,
            "gan_loss": gan_loss if gan else None,
            "D_loss": D_loss if gan else None,
            "fusion_image": fusion_image if fusion else None,
            "conditions": conditions_fornext,
            "w_predict": w_predict,
            "x_block_gen": x_block_gen,
            # 'time': [time2-time1,time3-time2,time4-time3,time5-time4,time5-time1],
        }

        return results

    def gen_image(
        self,
        seed=0,
        device="cuda:0",
        w_in=None,
        cs_in=None,
        batch_size=1,
        truncation_psi: float = 0.7,
        resolution: int = 512,
        conditions=None,
        prepare_data=True,
        res_plane=None,
        return_xblock=False,
        return_conditions=False,
    ):

        # 统一设备变量，优先从 self._get_device() 获取
        dev = device

        # ----------------------------------cam--------------------------
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
        c_front = torch.from_numpy(c_front).to(dev).unsqueeze(0).repeat(batch_size, 1)
        sh_ref_cam, intrinsics = decode_camera_params(c_front[0].cpu())

        # ------------------------------ w ---------------------------------
        if w_in != None:
            w = w_in
        else:
            rng = torch.Generator(dev)
            rng.manual_seed(seed)
            z = torch.randn(
                (batch_size, self.gghead._config.z_dim), device=dev, generator=rng
            )
            w = self.gghead.mapping(z, c_front, truncation_psi=truncation_psi)
            w = w.repeat(cs.shape[0] // w.shape[0], 1, 1)

        # ------------------------ image synthesis ---------------------------

        if return_xblock or return_conditions:
            x_block, output = self.gghead.synthesis(
                w,
                cs,
                sh_ref_cam=sh_ref_cam,
                noise_mode="const",
                neural_rendering_resolution=resolution,
                conditions=conditions,
                return_xblock=True,
                planes_res=res_plane,
            )
            for n, x in enumerate(x_block):
                x_block[n] = x[:batch_size]
            if return_conditions:

                conditions = {
                    "x_block": x_block,
                    "gs_model": clone_gaussian_model(self.gghead._gaussian_model),
                    "cs": cs,
                    "w": w,
                    "color_image": (output["image"][:batch_size] + 1) / 2,
                }
                return conditions
        else:
            output = self.gghead.synthesis(
                w,
                cs,
                sh_ref_cam=sh_ref_cam,
                noise_mode="const",
                neural_rendering_resolution=resolution,
                conditions=conditions,
                return_xblock=False,
                planes_res=res_plane,
            )
        if prepare_data:
            f_images = (output["image"][:batch_size] + 1) / 2
            t_images = (output["image"][batch_size:] + 1) / 2

            return (
                f_images,
                t_images,
                cs[:batch_size],
                cs[batch_size:],
                clone_gaussian_model(self.gghead._gaussian_model),
                x_block if return_xblock else None,
                w[:batch_size],
            )
        else:
            images = (output["image"][:batch_size] + 1) / 2
            return (
                images,
                clone_gaussian_model(self.gghead._gaussian_model),
                x_block if return_xblock else None,
            )

    def calc_metrics(self, results, gan=False, change=False, fusion=False):
        t_image = results["t_image"]
        gen_image = results["gen_image"]
        feedforward_image = results["feedforward_image"]
        adain_image = results["adain_image"]
        fusion_image = results["fusion_image"]

        pec_loss = self.percep_loss(gen_image, t_image)
        img_loss = self.loss_fn(gen_image, t_image)

        pec_feedforward_loss = self.percep_loss(feedforward_image, t_image)
        img_feedforward_loss = self.loss_fn(feedforward_image, t_image)

        pec_adain_loss = self.percep_loss(adain_image, t_image)
        img_adain_loss = self.loss_fn(adain_image, t_image)

        if fusion:
            pec_fusion_loss = self.percep_loss(fusion_image, t_image)
            img_fusion_loss = self.loss_fn(fusion_image, t_image)
            loss = {
                "img_loss": img_loss,
                "pec_loss": 0.02 * pec_loss,
                "img_feedforward_loss": img_feedforward_loss,
                "pec_feedforward_loss": 0.005 * pec_feedforward_loss,
                "img_adain_loss": img_adain_loss,
                "pec_adain_loss": 0.005 * pec_adain_loss,
                "img_fusion_loss": img_fusion_loss,
                "pec_fusion_loss": 0.02 * pec_fusion_loss,
            }
        else:
            loss = {
                "img_loss": img_loss,
                "pec_loss": 0.02 * pec_loss,
                "img_feedforward_loss": img_feedforward_loss,
                "pec_feedforward_loss": 0.005 * pec_feedforward_loss,
                "img_adain_loss": img_adain_loss,
                "pec_adain_loss": 0.005 * pec_adain_loss,
            }
        psnr = -10.0 * torch.log10(
            nn.functional.mse_loss(t_image, gen_image).detach() + 0.001
        )
        if gan:
            # if change:
            #     return results['gan_loss'], {"psnr": psnr.item()}, results['D_loss']
            return loss | results["gan_loss"], {"psnr": psnr.item()}, results["D_loss"]

        return loss, {"psnr": psnr.item()}

    def gs_gen(self, gs_attr_dict=None, c=None, bg=None, device=None, gs_model=None):
        """Render a batch of Gaussian-splat images from a gs_attr_dict and a list/tensor of cameras.

        Args:
            gs_attr_dict (dict): 包含键 'xyz','scaling','rotation','opacity','shs' 的高斯属性字典，
                                 形状为 [B, G, ...] 的张量集合。
            cam (iterable or tensor): 每帧对应的相机参数（与 gs_attr_dict 第一维一致）。
            bg (tensor or None): 可选背景颜色/图像；为 None 时使用 `self._gaussian_bg_train`。

        Returns:
            torch.Tensor: shape [B, C, H, W] 的渲染图像张量。
        """

        # 统一设备来源
        dev = device

        cam = c2cam(c)

        gen_images = []

        # Prepare a reference camera for SH -> RGB conversion
        c_front = encode_camera_params(
            Pose(
                matrix_or_rotation=np.eye(3),
                translation=(0, 0.0, 3.5),
                pose_type=PoseType.CAM_2_WORLD,
                camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL,
            ),
            DEFAULT_INTRINSICS,
        )
        c_front = torch.from_numpy(c_front).to(dev).unsqueeze(0)
        sh_ref_cam, intrinsics = decode_camera_params(c_front[0].cpu())
        intrinsics = intrinsics.rescale(512, inplace=False)
        gaussian_sh_ref_cam = pose_to_rendercam(
            sh_ref_cam, intrinsics, 512, 512, device=dev
        )

        sh_degree = self.gghead._config.gaussian_attribute_config.sh_degree
        n_feature_channels = (
            self.gghead._config.gaussian_attribute_config.n_color_channels
        )
        if gs_attr_dict:
            bsize = gs_attr_dict["xyz"].shape[0]
            for i in range(bsize):
                # assign attributes to the gaussian model
                self._gaussian_model._xyz = gs_attr_dict["xyz"][i].contiguous()
                self._gaussian_model._scaling = gs_attr_dict["scaling"][i].contiguous()
                self._gaussian_model._rotation = gs_attr_dict["rotation"][
                    i
                ].contiguous()
                self._gaussian_model._opacity = gs_attr_dict["opacity"][i].contiguous()
                self._gaussian_model._features_dc = gs_attr_dict["shs"][i][
                    :, [0]
                ].contiguous()
                self._gaussian_model._features_rest = gs_attr_dict["shs"][i][
                    :, 1:
                ].contiguous()

                # compute view-dependent RGB from SH coefficients
                shs_view = self._gaussian_model.get_features.view(
                    -1, (sh_degree + 1) ** 2, n_feature_channels
                ).permute(0, 2, 1)
                dir_pp = (
                    self._gaussian_model.get_xyz
                    - gaussian_sh_ref_cam.camera_center.repeat(1, 1)
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
                sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
                colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
                override_color = colors

                # select background
                bg_input = bg if bg is not None else self.gaussian_bg_train

                # render (some render() 返回 dict，有时直接返回 tensor)
                rendered = render(
                    cam[i],
                    self._gaussian_model,
                    PipelineParams2(),
                    bg_input,
                    override_color=override_color,
                )
                rendered_image = (
                    rendered["render"]
                    if isinstance(rendered, dict) and "render" in rendered
                    else rendered
                )

                gen_images.append(rendered_image)
        elif gs_model:
            # self._gaussian_model = gs_model
            shs_view = gs_model.get_features.view(
                -1, (sh_degree + 1) ** 2, n_feature_channels
            ).permute(0, 2, 1)
            dir_pp = (
                gs_model.get_xyz
                - gaussian_sh_ref_cam.camera_center.repeat(1, 1)
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
            sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
            colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
            override_color = colors

            # select background
            bg_input = bg if bg is not None else self.gaussian_bg_train

            # render (some render() 返回 dict，有时直接返回 tensor)
            rendered = render(
                cam[0],
                gs_model,
                PipelineParams2(),
                bg_input,
                override_color=override_color,
            )
            rendered_image = (
                rendered["render"]
                if isinstance(rendered, dict) and "render" in rendered
                else rendered
            )

            gen_images.append(rendered_image)

        gen_images = torch.stack(gen_images, 0)
        return gen_images

    def get_mask_index(
        self, gaussian_model, mask, cam, value=0.7, return_weights=False
    ):
        weights = torch.zeros_like(gaussian_model._opacity)
        weights_cnt = torch.zeros_like(gaussian_model._opacity, dtype=torch.int32)
        # mask_test = random_block_mask(512, 512, 16).to(self.device)
        gaussian_model.apply_weights(cam, weights, weights_cnt, mask)
        weights_return = weights.clone()
        weights /= weights_cnt + 1e-7
        selected_mask = weights > value
        selected_mask = selected_mask[:, 0]
        if return_weights:
            return selected_mask, weights_return
        return selected_mask

    def get_mask_uv(
        self,
        gs,
        batch_size,
        cam,
        mask=None,
        value1=0.1,
        value2=0.1
    ):
        uv_mask_ = []
        uv_mask_more_ = []
        for i in range(batch_size):
            if mask == None:
                mask_, cam_ = (
                    random_irregular_mask_smooth(512, 512, area=[100, 2]).to(
                        self.device
                    ),
                    cam[i],
                )
            else:
                mask_, cam_ = mask[i], cam[i]
            
            # self._gaussian_model = gs

            mask_index = self.get_mask_index(
                gs, mask_, cam_, value=value1
            )
            # mask_index = ~mask_index
            coords = self.gghead._uv_idx[mask_index]

            uv_mask = torch.zeros((512, 512))
            uv_mask[coords[:, 1], coords[:, 0]] = 1.0
            uv_mask_.append(uv_mask)
            ###########################################################

            mask_gt = 1 - mask_
            # mask_gt = mask_
            mask_index = self.get_mask_index(
                gs, mask_gt, cam_, value=value2
            )  # 0.9
            mask_index = ~mask_index
            coords = self.gghead._uv_idx[mask_index]

            uv_mask = torch.zeros((512, 512))
            uv_mask[coords[:, 1], coords[:, 0]] = 1.0
            uv_mask_more_.append(uv_mask)

        return (
            (torch.stack(uv_mask_, 0)[:, None, ...]).bool().to(self.device),
            (torch.stack(uv_mask_more_, 0)[:, None, ...]).bool().to(self.device),
        )

    def mask2uv_mask(self, mask, batch_size, f_cam, gs_gt, gs):

        gt_uv_mask, gt_uv_mask_more = self.get_mask_uv(
            gs_gt,
            batch_size,
            f_cam,
            mask=mask,
        )
        
        # gt_uv_mask_res, _ = self.get_mask_uv(
        #     gs_gt,
        #     batch_size,
        #     f_cam,
        #     mask= 1-mask,
        #     value1=0.8
        # )

        gen_uv_mask, gen_uv_mask_more = self.get_mask_uv(gs, batch_size, f_cam,mask=mask)

        uv_mask_less = gt_uv_mask | gen_uv_mask
        uv_mask_more = (
            gt_uv_mask_more | gen_uv_mask_more
        )  # & (mask_screen_all)#| mask_less1

        # mask_gt_more = mask_gt | mask_gt_  # & (mask_screen_all)#| mask_less1
        # mask_gt_less = mask_less1|mask_less0
        # mask_gt_less = (mask_less1 | mask_less0) #& (~mask_res) # & mask_screen_all2
        uv_mask_less = dilate_mask_scipy(uv_mask_less, 25)
        uv_mask = uv_mask_less & uv_mask_more #& (~gt_uv_mask_res)
        
        return uv_mask

    def run_inversion_our(self, target_image_tensor):

        result = self.forward(
            batch_size=1,
            device=self.device,
            f_image=target_image_tensor,
            mask=None,
        )

        w_predict = result["w_predict"]
        conditions = result["conditions"]["conditions_gt"]
        single_image_tensor = target_image_tensor[0]
        single_image_numpy = (single_image_tensor.permute(1, 2, 0)).clamp(0, 1)
        single_image_numpy = (single_image_numpy.cpu().numpy() * 255).astype(np.uint8)

        # 调用 PTI.train
        result = self.projector.project_gaussian(
            single_image_numpy,
            verbose=True,
            cs=None,
            w_start=w_predict,
            conditions=conditions,
        )

        # 提取重建结果
        recon_tensor = result["image"]

        if recon_tensor.ndim == 3:
            recon_tensor = recon_tensor.unsqueeze(0)
        
        
        conditions = self.gen_image(
            device=result["w"].device,
            conditions=conditions,
            w_in=result["w"],
            return_conditions=True,
            batch_size=1,
            prepare_data=False,    
            cs_in=result["c"].to(self.device), 
        )
        conditions["color_image"] = target_image_tensor
        # self.get_video(result["c"].to(self.device))
        return conditions, result

    def get_video(self, c):
        cs_list = get_cs_list(c, device=self.device)
        from dreifus.image import Img
        pictures = self.gs_gen(self, gs_attr_dict=None, c=cs_list, bg=None, device=self.device, gs_model=self.gghead._gaussian_model)
        frames = [Img.from_normalized_torch(image).to_numpy().img[..., :3] for image in pictures]
        from elias.util import ensure_directory_exists
        output_folder = f"output/sampled_heads/"
        ensure_directory_exists(output_folder)
        import mediapy
        seed = random.randint(0, 10000)
        mediapy.write_video(f"{output_folder}/{seed:04d}.mp4", frames, fps=16)