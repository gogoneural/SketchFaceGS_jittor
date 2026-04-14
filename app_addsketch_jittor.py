import os
os.environ["JT_SYNC"] = "1"
os.environ["trace_py_var"] = "3"
import sys
import faulthandler
import signal
import traceback

# Enable faulthandler to print Python stack trace on segfault
faulthandler.enable()

def _debug_segfault_handler(signum, frame):
    """Print all thread stacks when SIGSEGV/SIGABRT is caught."""
    sys.stderr.write(f"\n{'='*60}\n[DEBUG-SEGFAULT] Signal {signum} caught!\n{'='*60}\n")
    sys.stderr.write(f"[DEBUG-SEGFAULT] Faulting thread: {threading.current_thread().name} (tid={threading.get_ident()})\n")
    try:
        worker = getattr(pipeline, '_jt_worker', None)
        if worker:
            sys.stderr.write(f"[DEBUG-SEGFAULT] JittorWorker thread alive: {worker._thread.is_alive()}, tid={worker._thread.ident}\n")
    except Exception:
        pass
    sys.stderr.write(f"[DEBUG-SEGFAULT] All threads:\n")
    for tid, tframe in sys._current_frames().items():
        sys.stderr.write(f"\n--- Thread {tid} ---\n")
        traceback.print_stack(tframe, file=sys.stderr)
    sys.stderr.flush()
    # Re-raise to get core dump
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)

# Register after threading is imported (deferred below)
# 设置环境变量，确保在导入复杂库之前生效
os.environ.setdefault("GRADIO_TEMP_DIR", "./gradio_tmp")
_base_dir = os.path.dirname(os.path.abspath(__file__))
_third_party_path = os.path.join(_base_dir, "third_party")
_gghead_src = os.path.join(_third_party_path, "gghead-master", "gghead-master", "src")
_lhm_path = os.path.join(_third_party_path, "LHM")
_jgaussian_path = os.path.join(_third_party_path, "JGaussian-main")
# JGaussian-main MUST be last in this list so it ends up first in sys.path (insert(0,...))
# This ensures JGaussian's utils/ takes precedence over LHM's utils/
for _p in [_third_party_path, _lhm_path, _gghead_src, _jgaussian_path]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ==========================================
# 2. Standard Library Imports
# ==========================================
import math
import time
import random
import hashlib
import threading
import queue
from datetime import datetime

# Now register segfault debug handler (threading is available)
signal.signal(signal.SIGSEGV, _debug_segfault_handler)
signal.signal(signal.SIGABRT, _debug_segfault_handler)


_SKIP = object()  # sentinel returned by try_submit when worker is busy

class JittorWorker:
    """
    Jittor C++ 后端不支持多线程调用。
    所有 Jittor/CUDA 操作必须在同一个 OS 线程上执行。
    其他线程通过 submit() 提交任务并等待结果。
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._busy = threading.Lock()  # held while worker is executing a task
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="JittorWorker"
        )
        self._thread.start()

    def _run(self):
        import jittor as jt
        _tid = threading.get_ident()
        print(f"[JittorWorker] started on OS thread {_tid}", flush=True)
        while True:
            task = self._queue.get()
            if task is None:
                break
            fn, args, kwargs, event, holder = task
            _fname = getattr(fn, '__name__', repr(fn))
            print(f"[JittorWorker] >>> exec {_fname} on tid={_tid}", flush=True)
            with self._busy:
                try:
                    holder["value"] = fn(*args, **kwargs)
                except Exception as e:
                    holder["error"] = e
                # Barrier: ensure ALL async CUDA ops finish before releasing
                try:
                    jt.sync_all()
                except Exception:
                    pass
            print(f"[JittorWorker] <<< done {_fname}", flush=True)
            event.set()

    @property
    def is_busy(self):
        """True if the worker is currently executing a task or has pending tasks."""
        return self._busy.locked() or not self._queue.empty()

    def submit(self, fn, *args, **kwargs):
        """在 Jittor 专用线程上执行 fn，阻塞直到完成。"""
        if threading.current_thread() is self._thread:
            return fn(*args, **kwargs)
        event = threading.Event()
        holder = {}
        self._queue.put((fn, args, kwargs, event, holder))
        event.wait()
        if "error" in holder:
            raise holder["error"]
        return holder["value"]

    def try_submit(self, fn, *args, **kwargs):
        """Submit if worker is idle, otherwise return _SKIP sentinel immediately.
        Used by preview renders to drop frames instead of blocking."""
        if self.is_busy:
            return _SKIP
        return self.submit(fn, *args, **kwargs)


# ==========================================
# 3. Third-Party Library Imports
# ==========================================
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import gradio_client.utils as gradio_client_utils
from gradio_imageslider import ImageSlider

if not getattr(gradio_client_utils, "_sketchfacegs_bool_schema_patch", False):
    _original_json_schema_to_python_type = gradio_client_utils._json_schema_to_python_type

    def _patched_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return _original_json_schema_to_python_type(schema, defs)

    gradio_client_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
    gradio_client_utils._sketchfacegs_bool_schema_patch = True

# ==========================================
# 4. Jittor Imports (替换 PyTorch)
# ==========================================
import jittor as jt
from jittor import nn
from jittor import transform as transforms
jt.flags.use_cuda = 1 if jt.has_cuda else 0
jt.flags.use_parallel_op_compiler = 0  # disable parallel C++ compilation threads
import pickle

# ==========================================
# 5. Project Specific Imports
# ==========================================
# Model related
from src.models.model_jittor import SketchFaceGS, encode_camera_params, decode_camera_params
from src.models.libs.sketch_model.create_sketch import SketchSimplifier
from src.utils.gaussian_utils import _apply_opacity_activation, _apply_color_activation
# Utils & Helpers
from src.utils.camera_utils import get_angles_from_camera_params, rand_c2w, c2cam, LookAtPoseSampler
from src.utils.utils import load_config
from src.utils.gaussian_utils import clone_gaussian_model


def _detach_conditions(cond):
    """Deep clone+stop_grad ALL Jittor Vars in a conditions dict.

    Severs ALL references to the forward-pass computation graph so Python GC
    can free old ops.  Prevents lived_ops accumulation -> fused_op.cc crash.
    """
    if not isinstance(cond, dict):
        return cond
    out = {}
    for k, v in cond.items():
        if isinstance(v, jt.Var):
            out[k] = v.clone().stop_grad()
        elif isinstance(v, (list, tuple)):
            out[k] = type(v)(
                x.clone().stop_grad() if isinstance(x, jt.Var) else x for x in v
            )
        else:
            out[k] = v
    return out


# Import bgrm directly, bypassing face_utils/__init__.py which imports torch-dependent FLAME
import importlib.util as _ilu
_bgrm_path = os.path.join(
    _third_party_path, "gghead-master", "gghead-master", "src",
    "gghead_jittor", "face_utils", "bgrm.py"
)
_bgrm_spec = _ilu.spec_from_file_location("gghead_jittor.face_utils.bgrm", _bgrm_path)
_bgrm = _ilu.module_from_spec(_bgrm_spec)
_bgrm_spec.loader.exec_module(_bgrm)
modnet = _bgrm.modnet
mask_image = _bgrm.mask_image

from gghead_jittor.constants import DEFAULT_INTRINSICS

# 3D/Rendering Libraries (dreifus_compat, JGaussian)
from src.utils.dreifus_compat import Pose, Intrinsics, CameraCoordinateConvention, PoseType
from src.utils.jgaussian_compat import PipelineParams2, pose_to_rendercam
from utils.sh_utils import eval_sh
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel


RT_SWITCH = False

import tempfile  # 记得在头部引入

cam_pivot = [0, 0.0, 0.175]

# 储存jpg加速传输
# def save_as_jpeg(pil_image, quality=40):
#     """
#     将 PIL 图片压缩为 JPEG 并保存为临时文件，返回文件路径。
#     解决 Base64 字符串过长导致的 OSError [Errno 36]。
#     """
#     # JPEG 不支持透明通道，需转 RGB
#     if pil_image.mode == "RGBA":
#         pil_image = pil_image.convert("RGB")

#     # 创建一个临时文件，关闭自动删除，否则 Gradio 读取不到
#     # suffix='.jpg' 很重要，Gradio 会根据后缀识别 MIME 类型
#     tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)

#     # 保存压缩后的图片
#     pil_image.save(tmp_file.name, format="JPEG", quality=quality)

#     # 关闭文件句柄（Windows下必须关闭才能被其他进程读取，Linux下是个好习惯）
#     tmp_file.close()

#     # 返回文件路径，例如 "/tmp/tmp8a7s_d9s.jpg"
#     return tmp_file.name
def save_as_jpeg(pil_image, quality=40):
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    
    # 强制指定目录为你设置的 gradio_tmp
    output_dir = "./gradio_tmp"
    os.makedirs(output_dir, exist_ok=True) # 确保文件夹存在
    
    # 传入 dir 参数
    tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=output_dir)

    pil_image.save(tmp_file.name, format="JPEG", quality=quality)
    tmp_file.close()
    return tmp_file.name

import cv2
import numpy as np
from PIL import Image

import numpy as np
from PIL import Image


# 把线稿编辑后的4通道图片变成三通道sketch
def extract_sketch_from_rgba(layer_np):

    # layer_np = np.array(layer_pil)

    # 4. 提取线条逻辑
    # 获取 Alpha 通道 (0=全透明背景, >0=有笔触)
    alpha_channel = layer_np[:, :, 3]

    # 创建一张全白底图
    sketch_result = np.ones((512, 512, 3), dtype=np.uint8) * 255

    # 定义阈值：只要 Alpha 大于一定值 (比如 5)，就认为是线条
    # 将这些位置设为纯黑
    mask = alpha_channel > 2
    sketch_result[mask] = [0, 0, 0]

    return sketch_result
  
  
def extract_sketch_from_composite(composite_np, background_np, diff_thresh=8):
    if composite_np is None or background_np is None:
        return None

    if composite_np.ndim == 2:
        composite_rgb = cv2.cvtColor(composite_np, cv2.COLOR_GRAY2RGB)
    else:
        composite_rgb = composite_np[:, :, :3]

    if background_np.ndim == 2:
        background_rgb = cv2.cvtColor(background_np, cv2.COLOR_GRAY2RGB)
    else:
        background_rgb = background_np[:, :, :3]

    if composite_rgb.shape[:2] != (512, 512):
        composite_rgb = cv2.resize(composite_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    if background_rgb.shape[:2] != (512, 512):
        background_rgb = cv2.resize(background_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)

    gray_bg = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2GRAY)
    gray_comp = cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray_bg, gray_comp)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    sketch = np.ones((512, 512, 3), dtype=np.uint8) * 255
    sketch[mask > 0] = [0, 0, 0]
    return sketch

def normalize_editor_image(img, target_size=512):
    if img is None:
        return np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    if isinstance(img, str):
        img = Image.open(img)

    img = np.array(img)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        white = np.ones_like(rgb) * 255.0
        img = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]

    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return img


def extract_editor_sketch(img, target_size=512):
    if img is None:
        return np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    if isinstance(img, dict):
        layers = img.get("layers") or []
        if len(layers) > 0 and layers[0] is not None:
            layer_np = np.array(layers[0])
            if layer_np.shape[:2] != (target_size, target_size):
                layer_np = cv2.resize(layer_np, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            if layer_np.ndim == 3 and layer_np.shape[2] == 4 and np.any(layer_np[:, :, 3] > 2):
                return extract_sketch_from_rgba(layer_np)

        composite = img.get("composite")
        background = img.get("background")
        if composite is not None and background is not None:
            composite_np = normalize_editor_image(composite, target_size=target_size)
            background_np = normalize_editor_image(background, target_size=target_size)
            extracted = extract_sketch_from_composite(composite_np, background_np)
            if extracted is not None:
                return extracted

        if composite is not None:
            return normalize_editor_image(composite, target_size=target_size)
        if background is not None:
            return normalize_editor_image(background, target_size=target_size)

    return normalize_editor_image(img, target_size=target_size)

# 跟上一个相反，4通道显示变成3通道sketch
def get_layer_from_sketch(sketch_numpy, ratio=0.6, target_size=512):
    if sketch_numpy.ndim == 3:
        sketch_gray = sketch_numpy[:, :, 0]
    else:
        sketch_gray = sketch_numpy

    # Resize 线稿
    if sketch_gray.shape[:2] != (target_size, target_size):
        sketch_gray = cv2.resize(
            sketch_gray, (target_size, target_size), interpolation=cv2.INTER_LINEAR
        )

    # ----------------------------------------------------
    # 2. 制作线稿图层：随 ratio 显形
    # ----------------------------------------------------

    # A. 提取线条区域 (假设输入线稿是 白底黑线)
    # 阈值处理：低于 200 的像素被认为是线条 (变黑)
    # 结果 binary_mask 中，线条位置为 255 (白色)，背景为 0
    _, binary_mask = cv2.threshold(sketch_gray, 200, 255, cv2.THRESH_BINARY_INV)

    # (可选) 如果线条太细，可以取消注释下面的膨胀代码
    # kernel = np.ones((2, 2), np.uint8)
    # binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # B. 创建 RGBA 图层
    layer_rgba = np.zeros((target_size, target_size, 4), dtype=np.uint8)

    # 颜色通道 (RGB)：永远是纯黑 (0,0,0)
    layer_rgba[:, :, 0:3] = 0

    # Alpha 通道：由 ratio 决定
    # ratio = 0 -> alpha = 0 (全透明，看不见线条) -> 对应“完全是 RGB”
    # ratio = 1 -> alpha = 255 (全不透明，黑实线) -> 对应“完全是线稿”
    current_alpha = int(255 * ratio)

    # 只把线条区域 (binary_mask > 0) 的 Alpha 设为 current_alpha
    # 背景区域保持 alpha=0 (透明)
    layer_rgba[binary_mask > 0, 3] = current_alpha
    return layer_rgba


# sketch和rgb到显示（硫酸纸）
def build_sketch_on_rgb_output(
    rgb_image_pil, sketch_numpy, ratio=0.6, preview_mode=False, target_size=512
):
    # 1. 准备底图 (Resize)
    bg_raw = normalize_editor_image(rgb_image_pil, target_size=target_size)
  
    # ----------------------------------------------------
    # 1. 制作背景：随 ratio 变白
    # ----------------------------------------------------
    # ratio = 0 -> 100% bg_raw (RGB)
    # ratio = 1 -> 100% white_overlay (纯白)
    white_overlay = np.ones_like(bg_raw) * 255

    # 混合公式：RGB * (1 - ratio) + White * ratio
    bg_blended_np = cv2.addWeighted(bg_raw, 1.0 - ratio, white_overlay, ratio, 0)
    bg_blended_pil = Image.fromarray(bg_blended_np)
  
    # === 预览模式 (JPEG) ===
    if preview_mode or sketch_numpy is None:
        jpeg_path = save_as_jpeg(bg_blended_pil, quality=35)
        return {"background": jpeg_path, "layers": [], "composite": jpeg_path}

    # === 正常模式：处理线稿 ===
    # 确保线稿是单通道灰度

    layer_rgba = get_layer_from_sketch(
        sketch_numpy, ratio=ratio, target_size=target_size
    )
    layer_pil = Image.fromarray(layer_rgba)

    # ----------------------------------------------------
    # 3. 制作合成预览图 (Composite)
    # ----------------------------------------------------
    # 将半透明的线稿 叠在 变白的背景上

    comp_pil = bg_blended_pil.convert("RGBA")

    # 使用 alpha_composite 确保半透明叠加效果正确
    # 效果：底图越白，上面的线条越黑越实，完美符合“硫酸纸”渐变
    comp_pil.alpha_composite(layer_pil)

    composite_np = np.array(comp_pil.convert("RGB"))

    bg_path = save_as_jpeg(bg_blended_pil, quality=90)
    comp_path = save_as_jpeg(Image.fromarray(composite_np), quality=90)

    # 【重要】
    # 返回 layers 让前端拿到独立的线稿层。
    # 前端可以将 layers[0] 放在 background 之上。
    # 用户在前端编辑时，实际上是在编辑 layers[0] (或前端生成的画布)，背景不动。
    return {"background": bg_path, "layers": [layer_pil], "composite": comp_path}
  
  
# line到mask
def _find_closed_regions(line_mask: np.ndarray, close_ksize: int = 19) -> np.ndarray:
    """
    给定二值线条图 (线=255, 背景=0)，
    找出被线条完全包围的闭合区域，返回填充后的 mask (闭合内部=255, 其余=0)。
    
    步骤：
    1. 先强制二值化（防止中间灰度值干扰）
    2. 形态学闭操作弥合线条的微小断口
    3. 取反 + 加边框 + flood fill 检测闭合内部
    """
    # 强制二值化：>0 的都算线条
    _, binary = cv2.threshold(line_mask, 1, 255, cv2.THRESH_BINARY)
    
    # 形态学闭操作 (dilate → erode)，弥合线条中的小缝隙
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    closed_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    # 取反: 线=0, 背景=255
    inv = cv2.bitwise_not(closed_lines)
    # 加 1px 白色边框，保证所有外部区域从角落可达
    h, w = inv.shape
    padded = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    # 从 (0,0) flood fill，把外部区域全部填成 0
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 0)
    # 去掉边框，剩下的 255 就是闭合内部区域
    closed_regions = flood[1:h+1, 1:w+1]
    return closed_regions


def get_line_diff_mask(
    img1: np.ndarray,
    img2: np.ndarray,
    diff_thresh: int = 40,
    dilate_iter: int = 2,
    kernel_size: int = 20,
):

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    # 检测闭合区域：线条围成的封闭区域直接作为 mask，不需要膨胀
    closed_fill = _find_closed_regions(mask)

    # 非闭合的线条仍然走膨胀逻辑
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=dilate_iter)

    # 合并：闭合填充区域 ∪ 膨胀线条区域
    mask_combined = cv2.bitwise_or(mask_dilated, closed_fill)
    return mask_combined


def get_brush_mask_from_layer(
    layer_rgba: np.ndarray,
    original_sketch: np.ndarray,
    opacity: float,
    min_rgb_thresh: int = 3,
    alpha_drop_thresh: int = 30,
    dilate_size: int = 5,
    dilate_iters: int = 2,
    close_ksize: int = 19,
) -> np.ndarray:
    """
    从编辑器返回的RGBA图层中精准提取用户笔画的mask。

    原理：
    - 原始线稿图层用纯黑(0,0,0)渲染，用户画笔用深灰(8,8,8)
    - 通过检测图层中 RGB > 0 的像素，精准定位用户添加的笔画（即使与原线重叠）
    - 通过检测 alpha 降低的像素，精准定位用户擦除的笔画
    - 结合闭合区域检测，自动填充被线条围住的区域
    """
    h, w = layer_rgba.shape[:2]

    # --- 1. 检测用户添加的笔画 (RGB > 0，因为原始线稿是纯黑) ---
    r, g, b = layer_rgba[:, :, 0], layer_rgba[:, :, 1], layer_rgba[:, :, 2]
    alpha = layer_rgba[:, :, 3]
    max_rgb = np.maximum(np.maximum(r, g), b)
    addition_mask = ((alpha > 2) & (max_rgb > min_rgb_thresh)).astype(np.uint8) * 255

    # --- 2. 检测用户擦除的笔画 (alpha 显著降低) ---
    original_layer = get_layer_from_sketch(original_sketch, ratio=opacity)
    orig_alpha = original_layer[:, :, 3].astype(np.int16)
    edit_alpha = layer_rgba[:, :, 3].astype(np.int16)
    alpha_drop = orig_alpha - edit_alpha
    erasure_mask = (alpha_drop > alpha_drop_thresh).astype(np.uint8) * 255

    # --- 3. 合并添加 + 擦除 ---
    stroke_mask = cv2.bitwise_or(addition_mask, erasure_mask)

    # --- 4. 闭合区域检测 ---
    closed_fill = _find_closed_regions(stroke_mask, close_ksize=close_ksize)

    # --- 5. 对笔画 mask 做二次膨胀（等效覆盖半径约49px）---
    if dilate_size > 0:
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        stroke_mask = cv2.dilate(stroke_mask, kernel, iterations=dilate_iters)

    # --- 6. 最终：膨胀后的笔画 ∪ 闭合区域 ---
    final_mask = cv2.bitwise_or(stroke_mask, closed_fill)
    return final_mask


def spherical(alpha_deg: float, beta_deg: float, radius: float):
    alpha_rad = math.radians(alpha_deg) + math.pi / 2
    beta_rad = math.radians(beta_deg) + math.pi / 2
    poses = LookAtPoseSampler.sample(
        alpha_rad,
        beta_rad,
        jt.array(cam_pivot).float32(),
        radius=radius,
        horizontal_stddev=0,
        vertical_stddev=0,
        batch_size=1,
    ).reshape(-1, 16)
    return poses


# 度数到角度
def get_cam(alpha_deg, beta_deg, radius=2.7, device=None, c=None):
    if c is not None:
        c2w = c[:, :16]
    else:
        c2w = spherical(alpha_deg, beta_deg, radius)
    cam2world_matrix = c2w.reshape(4, 4)
    intrinsics_matrix = np.array(DEFAULT_INTRINSICS).reshape(3, 3)

    cam_2_world_pose = Pose(
        cam2world_matrix.detach().numpy() if isinstance(cam2world_matrix, jt.Var) else np.array(cam2world_matrix),
        pose_type=PoseType.CAM_2_WORLD,
        disable_rotation_check=True,
    )
    intrinsics = Intrinsics(intrinsics_matrix)
    intrinsics = intrinsics.rescale(512, inplace=False)
    gaussian_camera = pose_to_rendercam(
        cam_2_world_pose, intrinsics, 512, 512
    )
    return gaussian_camera


# 渲染点云
class RenderWorker(threading.Thread):

    def __init__(self, width=512, height=512, backend="egl", daemon=True):
        super().__init__(daemon=daemon)
        self.width = width
        self.height = height
        self.backend = backend
        self.task_q = queue.Queue()
        self.result_q = queue.Queue()
        self._stop_flag = False

        # GL 资源
        self.ctx = None
        self.prog = None
        self.vbo = None
        self.vbo_1 = None
        self.vao = None
        self.fbo = None

    def _init_gl(self):
        import moderngl

        self.ctx = moderngl.create_standalone_context(backend=self.backend)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec3 in_color;
                in float in_size;
                out vec3 v_color;
                uniform mat4 mvp;
                uniform float pointScale;
                void main() {
                    vec4 clip = mvp * vec4(in_pos, 1.0);
                    gl_Position = clip;

                    // 透视除法得到 NDC 后的深度近似
                    float dist = abs(clip.z / clip.w);

                    // 点大小 = 输入的基础大小 * 缩放因子 / 距离
                    gl_PointSize = in_size * dist / pointScale;

                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                out vec4 f_color;
                void main() {
                    vec2 p = gl_PointCoord * 2.0 - 1.0;
                    float r2 = dot(p, p);
                    if (r2 > 1.0) discard;
                    float z = sqrt(1.0 - r2);
                    vec3 n = normalize(vec3(p, z));
                    //light
                    //右 下 前
                    vec3 L = normalize(vec3(0.5,-1.0,1.0));
                    vec3 V = vec3(0.0, 0.0, 1.0);
                    vec3 H = normalize(L + V);
                    float ambient = 0.25;
                    float diff = max(dot(n, L), 0.0);
                    float spec = pow(max(dot(n, H), 0.0), 12.0) * 0.3;
                    vec3 color = v_color * (ambient + (1.0 - ambient) * diff) + spec;
                    color = pow(color, vec3(1.0/1.5));
                    f_color = vec4(color, 1.0);
                }
            """,
        )
        self.fbo = self.ctx.simple_framebuffer((self.width, self.height))

    def _upload_points(self, pts: np.ndarray, size: np.ndarray):
        if pts.shape[0] == 0:
            return  # nothing to upload yet
        colors = np.random.uniform(0.0, 1.0, (pts.shape[0], 3)).astype("f4")
        colors = np.ones((pts.shape[0], 3)).astype("f4") * 0.7
        data = np.hstack([pts.astype("f4", copy=False), colors])
        raw = data.tobytes()
        size = size.tobytes()
        if self.vbo:
            self.vbo.release()
        self.vbo = self.ctx.buffer(reserve=len(raw))

        if self.vbo_1:
            self.vbo_1.release()
        self.vbo_1 = self.ctx.buffer(reserve=len(size))
        self.vbo.write(raw)
        self.vbo_1.write(size)
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "3f 3f", "in_pos", "in_color"), (self.vbo_1, "1f", "in_size")],
        )

    def _render_frame(self, alpha, beta, radius, eye_offset=cam_pivot):
        if self.vao is None:
            raise RuntimeError("No VAO. Upload points before rendering.")
        gaussian_cam = get_cam(alpha, beta, radius)
        if hasattr(self, "condition"):
            self.condition["cam"] = [gaussian_cam]
        mvp = (
            np.diag([1, -1, 1, 1]).astype("f4")
            @ gaussian_cam.full_proj_transform.detach().numpy().T
        )
        mvp = mvp.T
        self.prog["mvp"].write(mvp.tobytes())
        self.prog["pointScale"].value = radius
        self.fbo.use()
        self.fbo.clear(1, 1, 1, 1)
        self.vao.render(mode=self.ctx.POINTS)
        raw = self.fbo.read(components=3, alignment=1)
        img = Image.frombytes(
            "RGB", (self.width, self.height), raw, "raw", "RGB", 0, -1
        )
        return np.array(img)

    def run(self):
        self._init_gl()
        while not self._stop_flag:
            try:
                task, payload = self.task_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if task == "UPLOAD":
                    self._upload_points(**payload)
                    self.result_q.put(("OK", None))
                elif task == "RENDER":
                    frame = self._render_frame(**payload)
                    self.result_q.put(("OK", frame))
                elif task == "STOP":
                    self._stop_flag = True
                    self.result_q.put(("OK", None))
                else:
                    self.result_q.put(("ERR", ValueError(f"Unknown task {task}")))
            except Exception as e:
                self.result_q.put(("ERR", e))

        for obj in (self.vbo, self.vbo_1, self.vao, self.fbo):
            try:
                if obj:
                    obj.release()
            except Exception:
                pass

    def sync_upload(self, pts: np.ndarray, size: np.ndarray):
        self.task_q.put(("UPLOAD", dict(pts=pts, size=size)))
        status, data = self.result_q.get()
        if status != "OK":
            raise data

    def sync_render(self, alpha, beta, radius):
        self.task_q.put(("RENDER", dict(alpha=alpha, beta=beta, radius=radius)))
        status, data = self.result_q.get()
        if status != "OK":
            raise data
        return data

    def stop(self):
        self.task_q.put(("STOP", None))
        status, _ = self.result_q.get()


# ---- Checkpoint loader (Jittor, no torch needed) ----
def _remap_ckpt_key(k):
    return (
        k.replace(".attn.norm_added_q.", ".attn.norm_q.")
         .replace(".attn.norm_added_k.", ".attn.norm_k.")
         .replace(".ff.net.0.proj.", ".ff.net.0.0.")
         .replace(".ff_context.net.0.proj.", ".ff_context.net.0.0.")
    )

def _load_jittor_checkpoint(model, ckpt_path):
    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        state_np = pickle.load(f)
    model_sd = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in state_np.items():
        if v is None:
            skipped += 1
            continue
        target_k = k if k in model_sd else _remap_ckpt_key(k)
        if target_k in model_sd:
            try:
                param = model_sd[target_k]
                arr = jt.array(v) if not isinstance(v, jt.Var) else v
                if param.shape == arr.shape:
                    param.update(arr)
                    loaded += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        else:
            parts = k.split(".")
            obj = model
            try:
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                setattr(obj, parts[-1], jt.array(v) if not isinstance(v, jt.Var) else v)
                loaded += 1
            except (AttributeError, TypeError):
                skipped += 1
    print(f"  Loaded {loaded} params, skipped {skipped}")
    return model


# ================== 主业务：AI + Gaussian + GL Worker 管理 ==================
class AISketchAndColorTo3D:
    def __init__(self, checkpoint_path=None, config_path=None, gl_width=512, gl_height=512):
        self.model = None
        self.gl_worker = None
        self.gl_width = gl_width
        self.gl_height = gl_height
        self.alpha = 0
        self.beta = 0
        self.opacity = 0.7
        self.radius = 2.7
        self.SketchSimplifier = None  # initialized on JittorWorker thread
        self.c = None
        self.sketch_mask_dilate_size = 20
        self.sketch_mask_dilate_iters = 2
        self.display_sketch = None
        self._state_version = 0  # 状态版本号，用于同步定时器事件
        self.active_tab = "model3"  # Tab 隔离：防止 model3/model4 互相污染
        self._jt_worker = JittorWorker()  # 所有 Jittor/CUDA 操作在此线程上执行
        self._action_lock = threading.RLock()
        self._edit_inflight = threading.Event()
        self._edit_guard_until = 0.0
        self._edit_guard_seconds = 0.8
        self._last_applied_edit_signature = None
        self.last_ellipsoid_image = None
        self._view_anchor_sketch = None

        def _init_on_jt_thread():
            """在 JittorWorker 线程上完成所有 Jittor 初始化"""
            if checkpoint_path and os.path.exists(checkpoint_path):
                cfg = load_config(config_path)
                edit_cfg = getattr(cfg, 'EDIT', None)
                if edit_cfg:
                    self.sketch_mask_dilate_size = getattr(edit_cfg, 'sketch_mask_dilate_size', 20)
                    self.sketch_mask_dilate_iters = getattr(edit_cfg, 'sketch_mask_dilate_iters', 2)
                t0 = time.time()
                model = SketchFaceGS(model_cfg=cfg.MODEL, edit_cfg=edit_cfg)
                model = _load_jittor_checkpoint(model, checkpoint_path)
                model.eval()
                jt.gc()
                t1 = time.time()
                print(f"Model init time: {t1 - t0:.2f}s")

                self.model = model
                self.preprocess = transforms.Compose(
                    [
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                    ]
                )
                # Warmup
                try:
                    print("Warming up synthesis kernels...")
                    _cs_warm = rand_c2w(jt.array(cam_pivot).float32(), 1)
                    with jt.no_grad():
                        _ = model.gen_image(seed=0, return_conditions=True, batch_size=1,
                                            prepare_data=False, cs_in=_cs_warm)
                    jt.sync_all()
                    jt.gc()
                    del _cs_warm, _
                    print("Warmup complete.")
                except Exception as _e:
                    print(f"Warmup skipped ({_e})")

            self.SketchSimplifier = SketchSimplifier()
            self._gaussian_model_forbackground = None
            self._gaussian_model = GaussianModel(sh_degree=1)
            self._gaussian_model.active_sh_degree = 1
            self._gaussian_model.opacity_activation = _apply_opacity_activation

            from src.models.libs.sketch_model.networks_sketch import define_S
            self.SketchGen = define_S(3, 3, 64, "sketch_no_part", 'instance', not False, 'normal', 0.02, )
            _sketchgen_pkl = os.path.join(_base_dir, "assets", "sketchgen_numpy.pkl")
            if os.path.exists(_sketchgen_pkl):
                with open(_sketchgen_pkl, "rb") as f:
                    sg_weights = pickle.load(f)
                sg_sd = self.SketchGen.state_dict()
                for k, v in sg_weights.items():
                    if k in sg_sd and sg_sd[k].shape == np.array(v).shape:
                        sg_sd[k].update(jt.array(v))
                print(f"SketchGen loaded from {_sketchgen_pkl}")
            for param in self.SketchGen.parameters():
                param.requires_grad = False
            self._sh_ref_cam = None

            try:
                print("Warming up realtime edit/render kernels...")
                _cs_warm = rand_c2w(jt.array(cam_pivot).float32(), 1)
                with jt.no_grad():
                    _warm_cond = model.gen_image(
                        seed=0,
                        return_conditions=True,
                        batch_size=1,
                        prepare_data=False,
                        cs_in=_cs_warm,
                    )
                    _warm_sketch_input = jt.ones((1, 3, 512, 512)).float32()
                    _warm_mask = jt.zeros((1, 1, 512, 512)).float32()
                    _warm_edit = model(
                        batch_size=1,
                        sketch_img=_warm_sketch_input,
                        mask=_warm_mask,
                        conditions_gt=_warm_cond,
                        fusion=True,
                        cs_in=_cs_warm,
                    )
                    _warm_render = model.gs_gen(gs_model=_warm_cond["gs_model"], c=_cs_warm)
                    _warm_sketch = (self.SketchGen(_warm_render * 2 - 1) + 1) / 2
                    _warm_simplified = self.SketchSimplifier(_warm_sketch[:, [0]], get_sketch=False)
                jt.sync_all()
                del _cs_warm, _warm_cond, _warm_sketch_input, _warm_mask, _warm_edit, _warm_render, _warm_sketch, _warm_simplified
                print("Realtime warmup complete.")
            except Exception as _e:
                print(f"Realtime warmup skipped ({_e})")

        self._jt_worker.submit(_init_on_jt_thread)

        # GL worker init (not Jittor, safe on any thread)
        self._ensure_gl_worker()
        pts, size = self._jt_worker.submit(self.handle_pts)
        self._ensure_gl_worker()
        self.gl_worker.sync_upload(pts, size)
    def reset_cam(self):
        self.alpha = 0
        self.beta = 0
        self.radius = 2.7

    def _begin_edit_guard(self):
        self._edit_inflight.set()

    def _end_edit_guard(self):
        self._edit_inflight.clear()
        self._edit_guard_until = time.time() + self._edit_guard_seconds

    def _rotation_should_use_cached(self):
        return self._edit_inflight.is_set() or time.time() < self._edit_guard_until

    def _begin_action(self, blocking=True):
        return self._action_lock.acquire(blocking=blocking)

    def _end_action(self):
        self._action_lock.release()

    def _ensure_gl_worker(self):
        if self.gl_worker is None:
            self.gl_worker = RenderWorker(width=self.gl_width, height=self.gl_height)
            self.gl_worker.start()

    # 得到点云
    def handle_pts(self):
        xyz = self._gaussian_model._xyz
        if xyz is None or (hasattr(xyz, 'shape') and xyz.shape[0] == 0):
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
        pts = xyz.detach().numpy()
        opacity = self._gaussian_model.get_opacity.detach().numpy()
        size = self._gaussian_model.get_scaling.detach().numpy()
        if size.ndim == 1:
            size = size[:, None]
        size = size.min(1, keepdims=True)
        upper_thresh = np.percentile(size, 90)
        selected = (opacity > 0.2) & (size < upper_thresh)
        indices = selected.squeeze(1)
        pts = pts[indices]

        # print(f"[PointCloud] points: {pts.shape}")
        size = size[indices]
        size = size + 12.5
        return pts, size

    # 纯生成，生成
    def generate_with_input_generate_gaussian_model(
        self,
        sketch_pack,
        color_ref_numpy,
        alpha=0,
        beta=0,
        radius=2.7,
        progress=gr.Progress(track_tqdm=True),
    ):
        # self.reset_cam()
        self.alpha = alpha
        self.beta = beta
        self.radius = radius
        if sketch_pack is None or color_ref_numpy is None:
            raise gr.Error("Please Upload Sketch and Image")

        sketch_numpy = extract_editor_sketch(sketch_pack, target_size=512)

        if self.model is None:
            raise gr.Error("模型未加载。")

        def _jt_gen_from_sketch():
            sketch_tensor = self._to_tensor(sketch_numpy)
            color_ref_tensor = self._to_tensor(color_ref_numpy)
            with jt.no_grad():
                _ = self.model(
                    batch_size=1,
                    sketch_img=sketch_tensor,
                    f_image=color_ref_tensor,
                )
            jt.sync_all()
            self._gaussian_model = clone_gaussian_model(self.model.gghead._gaussian_model)
            return self.handle_pts()

        pts, size = self._jt_worker.submit(_jt_gen_from_sketch)
        self._ensure_gl_worker()
        self.gl_worker.sync_upload(pts, size)
        return self.render_from_cached_model(
            alpha=self.alpha,
            beta=self.beta,
            radius=self.radius,
            return_sketch=False,
            gen_sketch=False,
        )
        # return [rendered_image_numpy, ellipsoid_image]

    # 编辑的我们的方法生成的结果的  生成????没用还有问题
    def edit_with_input_generate_gaussian_model(
        self, edit_sketch, color_ref_numpy, progress=gr.Progress(track_tqdm=True)
    ):
        if not isinstance(edit_sketch["composite"], np.ndarray):
            edit_sketch = np.array(edit_sketch["composite"])[:, :, :3]
        else:
            edit_sketch = edit_sketch["composite"][:, :, :3]

        self.reset_cam()
        if edit_sketch is None or color_ref_numpy is None:
            raise gr.Error("Please Upload Sketch and Image")

        if not isinstance(color_ref_numpy, np.ndarray):
            color_ref_numpy = np.array(color_ref_numpy)

        if self.model is None:
            raise gr.Error("模型未加载。")

        def _jt_edit_from_sketch():
            sketch_tensor = self._to_tensor(edit_sketch)
            color_ref_tensor = self._to_tensor(color_ref_numpy)
            with jt.no_grad():
                results = self.model(
                    batch_size=1,
                    sketch_img=sketch_tensor,
                    f_image=color_ref_tensor,
                )
            jt.sync_all()
            self.condition = _detach_conditions(results["conditions"])
            if not (self.model and self.model._gaussian_model):
                raise gr.Error(
                    f"model: {self.model == None}, gaussian_model{self.model._gaussian_model == None}"
                )
            gaussian_cam = get_cam(self.alpha, self.beta, self.radius)
            if hasattr(self, "condition"):
                self.condition["cam"] = [gaussian_cam]
            rendered_image_tensor, depth = self._render_gaussian_locked(gaussian_cam)
            es = (self.SketchGen(rendered_image_tensor.unsqueeze(0) * 2 - 1) + 1) / 2
            es = self.SketchSimplifier(es.detach()[:, [0]], get_sketch=False)
            es = es.repeat(1, 3, 1, 1)
            self.last_sketch = (
                (es[0].detach().numpy() * 255)
                .astype(np.uint8)
                .transpose(1, 2, 0)
            )
            self.last_sketch_raw = self.last_sketch.copy()
            self.display_sketch = self.last_sketch.copy()
            self._view_anchor_sketch = self.last_sketch.copy()
            jt.sync_all()
            return self.handle_pts()

        self._begin_edit_guard()
        try:
            pts, size = self._jt_worker.submit(_jt_edit_from_sketch)
        finally:
            self._end_edit_guard()
        self._ensure_gl_worker()
        self.gl_worker.sync_upload(pts, size)
        return self.render_from_cached_model(
            alpha=self.alpha, beta=self.beta, radius=self.radius, return_sketch=False
        )

    # 再编辑
    def edit_gaussian_model(
        self,
        edit_sketch,
        alpha=0,
        beta=0,
        return_sketch=True,
        gen_sketch=True,
        keep_sketch=False,
        state_version=None,
    ):
        # 版本同步：定时器事件携带触发时的版本号，若与当前版本不匹配则为过期事件
        if state_version is not None and state_version != self._state_version:
            print(f"[Skip] Stale event (v{state_version} vs current v{self._state_version})")
            return _cached_mix_output(self.alpha, self.beta)
        print(f"{alpha} {self.alpha} {beta} {self.beta}")

        if alpha == self.alpha and beta == self.beta:
            anchor_sketch = self._view_anchor_sketch if self._view_anchor_sketch is not None else self.last_sketch

            # === 修改开始: 强力二值化提取 ===
            layers = edit_sketch.get("layers", [])

            if layers and len(layers) > 0 and layers[0] is not None:
                layer_rgba = np.array(layers[0])
                if layer_rgba.shape[:2] != (512, 512):
                    layer_rgba = cv2.resize(layer_rgba, (512, 512))
                edit_sketch_img = extract_sketch_from_rgba(layer_rgba)

                # 新方案：直接从图层提取用户笔画mask（画笔颜色#080808 vs 原始线稿#000000）
                mask_sketch = get_brush_mask_from_layer(
                    layer_rgba, anchor_sketch, self.opacity,
                    dilate_size=int(self.sketch_mask_dilate_size),
                    dilate_iters=int(self.sketch_mask_dilate_iters),
                )

            else:
                if keep_sketch:
                    # 定时器触发且无用户图层 → 没有用户笔画，跳过前馈
                    print("[Skip] No user-drawn layers (timer), skipping forward pass.")
                    return _cached_mix_output(self.alpha, self.beta)
                print("Warning: No layers found, falling back to diff method.")
                if not isinstance(edit_sketch["composite"], np.ndarray):
                    edit_sketch_img = np.array(edit_sketch["composite"])[:, :, :3]
                else:
                    edit_sketch_img = edit_sketch["composite"][:, :, :3]

                # 回退：使用旧的diff方案
                mask_sketch = get_line_diff_mask(
                    anchor_sketch, edit_sketch_img,
                    kernel_size=int(self.sketch_mask_dilate_size),
                    dilate_iter=int(self.sketch_mask_dilate_iters),
                )

            edit_sketch = edit_sketch_img

            # === 修改结束 ===
            if not keep_sketch:
                self.last_sketch = edit_sketch
                self.last_sketch_raw = edit_sketch.copy()
            

            # 如果mask全空（没有用户笔画），跳过前馈，避免模型漂移
            if mask_sketch.max() == 0:
                print("[Skip] Empty mask, skipping forward pass.")
                return _cached_mix_output(self.alpha, self.beta)

            edit_signature = (
                self.alpha,
                self.beta,
                hashlib.sha1(mask_sketch.tobytes()).hexdigest(),
                hashlib.sha1(edit_sketch.tobytes()).hexdigest(),
            )
            if keep_sketch and edit_signature == self._last_applied_edit_signature:
                print("[Skip] Duplicate realtime edit, skipping replay.")
                return _cached_mix_output(self.alpha, self.beta)

            def _jt_edit_forward():
                def _do_forward():
                    jt.sync_all()
                    _edit_sketch_t = self._to_tensor_raw(edit_sketch)
                    _mask_t = jt.array(mask_sketch[None, None, ...].astype(np.float32)) / 255
                    _cs_in = rand_c2w(jt.array(cam_pivot).float32(), 1, alpha_deg=self.alpha, beta_deg=self.beta)
                    jt.sync_all()
                    _t0 = time.time()
                    with jt.no_grad():
                        result = self.model(
                            batch_size=1,
                            sketch_img=_edit_sketch_t,
                            mask=_mask_t,
                            conditions_gt=self.condition,
                            fusion=True,
                            cs_in=_cs_in,
                        )
                    jt.sync_all()
                    print(f"[Timer] Edit forward pass: {(time.time() - _t0)*1000:.1f} ms")
                    cond = result["conditions"]
                    del result, _edit_sketch_t, _mask_t, _cs_in
                    self._gaussian_model = clone_gaussian_model(self.model.gghead._gaussian_model)
                    self.condition = _detach_conditions(cond)
                    del cond

                try:
                    _do_forward()
                except RuntimeError as e:
                    if "Unable to alloc" in str(e):
                        print("[OOM] CUDA OOM detected, running jt.gc() and retrying...", flush=True)
                        jt.sync_all()
                        jt.gc()
                        jt.sync_all()
                        _do_forward()
                    else:
                        raise
                jt.sync_all()
                return self.handle_pts()

            self._begin_edit_guard()
            try:
                pts, size = self._jt_worker.submit(_jt_edit_forward)
            finally:
                self._end_edit_guard()
            if keep_sketch:
                self.display_sketch = edit_sketch.copy()
                self._last_applied_edit_signature = edit_signature
            self._ensure_gl_worker()
            self.gl_worker.sync_upload(pts, size)

        return self.render_from_cached_model(
            alpha=self.alpha,
            beta=self.beta,
            radius=self.radius,
            return_sketch=return_sketch,
            gen_sketch=gen_sketch,
        )

    # gghead生成的 生成
    def edit_without_input_generate_gaussian_model(
        self, seed=None, progress=gr.Progress(track_tqdm=True)
    ):
        self._state_version += 1
        self.reset_cam()
        if self.model is None:
            raise gr.Error("模型未加载。")
        if seed is None or len(seed) == 0:
            seed = random.randint(0, 10000)
        else:
            seed = int(seed)

        def _jt_gen_model():
            with jt.no_grad():
                _raw_cond = self.model.gen_image(
                    seed=seed,
                    return_conditions=True,
                    batch_size=1,
                    prepare_data=False,
                    cs_in=rand_c2w(jt.array(cam_pivot).float32(), 1, alpha_deg=self.alpha, beta_deg=self.beta),
                )
            jt.sync_all()
            self.condition = _detach_conditions(_raw_cond)
            del _raw_cond
            self._gaussian_model = clone_gaussian_model(self.model.gghead._gaussian_model)
            self._gaussian_model_forbackground = clone_gaussian_model(self.model.gghead._gaussian_model)
            jt.sync_all()
            return self.handle_pts()

        self._begin_action(blocking=True)
        try:
            pts, size = self._jt_worker.submit(_jt_gen_model)
        finally:
            self._end_action()
        self._ensure_gl_worker()
        self.gl_worker.sync_upload(pts, size)
        return self.render_from_cached_model(
            alpha=self.alpha,
            beta=self.beta,
            radius=self.radius,
            return_sketch=True,
        ) + [seed]

    # 渲染全部
    def render_from_cached_model(
        self,
        alpha: float,
        beta: float,
        radius: float = 2.7,
        return_sketch=False,
        gen_sketch=True,
        mode="gen",
        sketch_forshow=None,
    ):
        # ... (前面的渲染逻辑完全保持不变) ...
        # ... 直到 rendered = Image.fromarray(rendered_image_numpy) 这一行 ...
        self.alpha = alpha
        self.beta = beta
        self.radius = radius
        if self.model is None or self.model._gaussian_model is None:
            # error_img = Image.new("RGB", (512, 512), color="black")
            from PIL import ImageDraw

            # draw = ImageDraw.Draw(error_img)
            # draw.text((150, 250), "请先点击 '生成3D模型'", fill="white")
            # arr = np.array(error_img)
            return arr, arr

        def _jt_render_cached():
            gaussian_cam = get_cam(self.alpha, self.beta, self.radius, c=self.c)
            if hasattr(self, "condition"):
                self.condition["cam"] = [gaussian_cam]
            rendered_image_tensor, depth = self._render_gaussian_locked(gaussian_cam)
            if not isinstance(rendered_image_tensor, jt.Var):
                raise TypeError("Gaussian render did not return jt.Var")
            if gen_sketch:
                angle_sketch = (
                    self.SketchGen(rendered_image_tensor.unsqueeze(0) * 2 - 1) + 1
                ) / 2
                angle_sketch = self.SketchSimplifier(
                    angle_sketch.detach()[:, [0]], get_sketch=False
                )
                angle_sketch = angle_sketch.repeat(1, 3, 1, 1)
                angle_sketch = (
                    (angle_sketch[0].detach().numpy() * 255)
                    .astype(np.uint8)
                    .transpose(1, 2, 0)
                )
                self.last_sketch_raw = angle_sketch.copy()
                self.last_sketch = extract_sketch_from_rgba(
                    get_layer_from_sketch(angle_sketch, ratio=self.opacity)
                )
                self._view_anchor_sketch = self.last_sketch.copy()
                self.display_sketch = self.last_sketch.copy()

            image_tensor_hwc = rendered_image_tensor.permute(1, 2, 0)
            image_tensor_255 = jt.clamp(image_tensor_hwc, 0.0, 1.0) * 255.0
            _rendered_np = image_tensor_255.uint8().numpy()

            _rendered_np2 = None
            if return_sketch:
                rendered_image_tensor2, _ = self._render_gaussian_locked(gaussian_cam, forbackground=True)
                image_tensor_hwc2 = rendered_image_tensor2.permute(1, 2, 0)
                image_tensor_255_2 = jt.clamp(image_tensor_hwc2, 0.0, 1.0) * 255.0
                _rendered_np2 = image_tensor_255_2.uint8().numpy()

            jt.sync_all()
            return _rendered_np, _rendered_np2

        self._begin_action(blocking=True)
        try:
            rendered_image_numpy, rendered_image_numpy2 = self._jt_worker.submit(_jt_render_cached)
        finally:
            self._end_action()

        ellipsoid_image = self.gl_worker.sync_render(self.alpha, self.beta, self.radius)
        rendered = Image.fromarray(rendered_image_numpy)
        ellipsoid_image = Image.fromarray(ellipsoid_image)

        # === 缓存当前的 RGB 渲染图，供滑块使用 ===
        self.last_rendered_image = rendered
        self.last_ellipsoid_image = ellipsoid_image

        if return_sketch:
            rendered2 = Image.fromarray(rendered_image_numpy2)
            self.last_rendered_image_back = rendered2
      
            editor_input = build_sketch_on_rgb_output(
                self.last_rendered_image_back, self.last_sketch, ratio=self.opacity, target_size=512
            )

            # else:
            #     editor_input = build_image_editor_input(self.last_sketch)

            return [
                [rendered, ellipsoid_image],
                editor_input,
            ]
        else:
            return rendered, ellipsoid_image

    def _get_sh_ref_cam(self):
        """Cache and return the SH reference camera (front-facing)."""
        if self._sh_ref_cam is not None:
            return self._sh_ref_cam
        c_front = encode_camera_params(
            Pose(
                matrix_or_rotation=np.eye(3),
                translation=(0, 0, 3.5),
                pose_type=PoseType.CAM_2_WORLD,
                camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL,
            ),
            Intrinsics(np.array(DEFAULT_INTRINSICS).reshape(3, 3)),
        )
        c_front_jt = jt.array(c_front).unsqueeze(0)
        sh_ref_cam, intrinsics = decode_camera_params(c_front_jt[0])
        intrinsics = intrinsics.rescale(512, inplace=False)
        self._sh_ref_cam = pose_to_rendercam(sh_ref_cam, intrinsics, 512, 512)
        return self._sh_ref_cam

    def _render_gaussian(self, cam, bg=None, forbackground=False):
        def _jt_fn():
            result = self._render_gaussian_locked(cam, bg=bg, forbackground=forbackground)
            jt.sync_all()
            return result
        return self._jt_worker.submit(_jt_fn)

    def _render_both_np(self, alpha, beta, radius, c=None, allow_drop=False):
        """Render main + background on JittorWorker thread, return numpy HWC uint8 images.
        If allow_drop=True and worker is busy, returns cached results (frame drop)."""
        def _jt_fn():
            cam = get_cam(alpha, beta, radius, c=c)
            main_t, depth_t = self._render_gaussian_locked(cam, forbackground=False)
            main_np = jt.clamp(main_t.permute(1, 2, 0), 0.0, 1.0).mul(255).uint8().numpy()
            if self._gaussian_model_forbackground is not None:
                bg_t, _ = self._render_gaussian_locked(cam, forbackground=True)
                bg_np = jt.clamp(bg_t.permute(1, 2, 0), 0.0, 1.0).mul(255).uint8().numpy()
            else:
                bg_np = main_np
            jt.sync_all()
            return main_np, bg_np

        if allow_drop:
            result = self._jt_worker.try_submit(_jt_fn)
            if result is _SKIP:
                # Worker busy – return cached preview
                cached = getattr(self, '_cached_both_np', None)
                if cached is not None:
                    return cached
                # No cache yet, must block
                result = self._jt_worker.submit(_jt_fn)
        else:
            result = self._jt_worker.submit(_jt_fn)

        self._cached_both_np = result
        return result

    def _render_gaussian_locked(self, cam, bg=None, forbackground=False):
        gaussian_sh_ref_cam = self._get_sh_ref_cam()

        sh_degree = self.model.gghead._config.gaussian_attribute_config.sh_degree
        n_feature_channels = (
            self.model.gghead._config.gaussian_attribute_config.n_color_channels
        )
        if forbackground:
            gaussian_model = self._gaussian_model_forbackground
        else:
            gaussian_model = self._gaussian_model
        shs_view = gaussian_model.get_features.reshape(
            -1, (sh_degree + 1) ** 2, n_feature_channels
        ).permute(0, 2, 1)

        dir_pp = (
            gaussian_model.get_xyz
            - gaussian_sh_ref_cam.camera_center.unsqueeze(0).broadcast(
                [gaussian_model.get_xyz.shape[0], 3]
            )
        )
        dir_pp_normalized = dir_pp / jt.norm(dir_pp, dim=-1, keepdim=True)
        sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
        colors = jt.clamp(sh2rgb + 0.5, min_v=0.0)

        rendered = render(
            cam,
            gaussian_model,
            PipelineParams2(),
            self.model.gaussian_bg_train,
            override_color=colors,
        )
        # Force all custom CUDA kernels (gaussian splatting) to complete
        jt.sync_all()

        return rendered["render"], rendered["depth"].repeat(3, 1, 1)

    def stop(self):
        if self.gl_worker:
            self.gl_worker.stop()
            self.gl_worker.join(timeout=2)
            # print("[Main] RenderWorker stopped cleanly.")

    # def save_other_angle(self, result):
    #     if self.c != None:
    #         c_3 = get_c3(self.c, 0.1, 0.5, self.device)

    #         gaussian_cam1 = get_cam(
    #             self.alpha, self.beta, self.radius, device=self.model.device, c=c_3[1]
    #         )
    #         gaussian_cam2 = get_cam(
    #             self.alpha, self.beta, self.radius, device=self.model.device, c=c_3[2]
    #         )
    #         gaussian_cam0 = get_cam(
    #             self.alpha, self.beta, self.radius, device=self.model.device, c=c_3[0]
    #         )
    #     else:

    #         gaussian_cam1 = get_cam(
    #             self.alpha - 0.1, self.beta - 0.5, self.radius, device=self.model.device
    #         )
    #         gaussian_cam2 = get_cam(
    #             self.alpha + 0.1, self.beta + 0.5, self.radius, device=self.model.device
    #         )
    #         gaussian_cam0 = get_cam(
    #             self.alpha, self.beta, self.radius, device=self.model.device
    #         )
    #     rendered_image_tensor, depth = self._render_gaussian(gaussian_cam1)
    #     rendered_image_tensor, depth = self._render_gaussian(gaussian_cam2)

    #     rendered_image_tensor_cat = self.model.gs_gen(
    #         result["gs_attr_dict_cat"], [gaussian_cam0]
    #     )
    #     rendered_image_tensor_zero = self.model.gs_gen(
    #         result["gs_attr_dict_zero"], [gaussian_cam0]
    #     )
    #     self.model.gs_gen(result["gs_attr_dict"], [gaussian_cam0])
    # 换一个gaussian_model
    def change_ply(self, path):
        def _jt_fn():
            self.model._gaussian_model.load_ply(path)
            return self.handle_pts()
        self._begin_action(blocking=True)
        try:
            pts, size = self._jt_worker.submit(_jt_fn)
        finally:
            self._end_action()
        self._ensure_gl_worker()
        self.gl_worker.sync_upload(pts, size)

    # 更新sketch透明度
    def update_sketch_opacity(self, opacity, sketch_edit=None):
        self.opacity = opacity
        # 检查是否有缓存的图片
        if not hasattr(self, "last_rendered_image") or self.last_rendered_image is None:
            return gr.update()

        current_sketch = self.display_sketch if self.display_sketch is not None else self.last_sketch

        # 1. 重新生成背景和图层数据
        # img_data 是一个字典: {"background": ..., "layers": [...], "composite": ...}
        # 这里复用了上一轮写好的逻辑：ratio 越大，背景越白，线条越显形
        img_data = build_sketch_on_rgb_output(
            self.last_rendered_image_back, current_sketch, ratio=opacity, target_size=512
        )

        # 2. 【关键修改】计算画笔的颜色
        # 逻辑统一：画笔的透明度必须等于线稿图层的透明度
        # 如果 opacity=1.0 -> alpha=255 (FF) -> 纯黑实线
        # 如果 opacity=0.5 -> alpha=128 (80) -> 半透明黑线
        alpha_val = int(255 * opacity)

        # 将数值转换为 2 位十六进制字符串 (例如: 255->'ff', 10->'0a')
        alpha_hex = f"{alpha_val:02x}"

        # 构造带透明度的 Hex 颜色代码: #RRGGBBAA
        # #000000 是黑色，后面接上 alpha_hex
        brush_color = f"#080808{alpha_hex}"

        # 3. 返回更新
        # 这样设置后，用户用画笔画出来的线条，会和当前显示的线稿层具有完全相同的“淡度”
        return gr.update(
            value=img_data, brush=gr.Brush(colors=[brush_color], color_mode="fixed")
        )

    def update_current_to_background(self):
        if self._gaussian_model is None:
            return gr.update()
        
        def _jt_fn():
            self._gaussian_model_forbackground = clone_gaussian_model(self._gaussian_model)
            jt.sync_all()
            gaussian_cam = get_cam(self.alpha, self.beta, self.radius, c=self.c)
            rendered_tensor, _ = self._render_gaussian_locked(gaussian_cam, forbackground=True)
            image_tensor_hwc = rendered_tensor.permute(1, 2, 0)
            image_tensor_255 = jt.clamp(image_tensor_hwc, 0.0, 1.0) * 255.0
            _np = image_tensor_255.uint8().numpy()
            jt.sync_all()
            return _np

        self._begin_action(blocking=True)
        try:
            rendered_image_numpy = self._jt_worker.submit(_jt_fn)
        finally:
            self._end_action()
        rendered_pil = Image.fromarray(rendered_image_numpy)
        
        # 3. 更新缓存
        self.last_rendered_image_back = rendered_pil
        current_sketch = self.display_sketch if self.display_sketch is not None else self.last_sketch
        
        # 4. 重新构建线稿编辑器的输入 (背景变了，线稿不变)
        editor_data = build_sketch_on_rgb_output(
            self.last_rendered_image_back, 
            current_sketch, 
            ratio=self.opacity, 
            target_size=512
        )
        
        return editor_data
    def mask_gaussian(self,gs_model,mask,cs): 
        mask_index = self.model.get_mask_index(
                    gs_model, mask[0], c2cam(cs)[0], value=0.4
                )
        target_slice = gs_model._features_dc[mask_index]

# 2. 创建一个形状相同但全为 0 的张量 (初始化为黑色 [0, 0, 0])
        light_blue_color = jt.array([0.5, 0.8, 1.0])

        # 3. 将最后一维的第三个通道（蓝色通道，索引为 2）设置为 1.0
        # 使用 [...] 可以适配不同维度的张量 (例如 (N, 3) 或 (N, 1, 3))
        # blue_features[..., 2] = 0.7

        # 4. 将构建好的蓝色特征赋值回原模型
        gs_model._features_dc[mask_index] = light_blue_color
        gs_model._features_rest[mask_index] = 0
        return gs_model

    # ---- Helper: numpy/PIL → Jittor tensor ----
    def _to_tensor(self, img):
        """Convert numpy/PIL image to [1, 3, H, W] Jittor tensor in [0, 1]."""
        t = self.preprocess(img)
        if not isinstance(t, jt.Var):
            t = jt.array(t)
        return t.unsqueeze(0)

    def _to_tensor_raw(self, img):
        """Convert numpy HWC uint8 image to [1, 3, H, W] Jittor tensor in [0, 1]."""
        if isinstance(img, np.ndarray):
            if img.shape[0] != 512 or img.shape[1] != 512:
                img = cv2.resize(img, (512, 512))
            t = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            return jt.array(t).unsqueeze(0)
        return self._to_tensor(img)

# resize
def handle_resize(img):
    return normalize_editor_image(img, target_size=512)

# mask掉rgb的背景
def handle_upload_appearance(img):
    img = np.array(img)
    if img.shape[0] != 512 or img.shape[1] != 512:
        img = cv2.resize(img, (512, 512))  # [HWC] 0-255
    mask = modnet(img[None, ...])[0]
    img = mask_image(img, mask) / 255
    return img


# 上传线稿变成展示的格式
def build_image_editor_output(img):
    """把用户上传的线稿转换成 layered dict（兼容你之前的逻辑）"""
    src_img = img.get("composite") if isinstance(img, dict) else img
    if src_img is None and isinstance(img, dict):
        src_img = img.get("background")
    src_img = normalize_editor_image(src_img, target_size=512)
    layer_channel = np.where(src_img != 255, 255 - src_img, 0)
    layer_channel = layer_channel[:, :, 0]
    out_layers = np.zeros((512, 512, 4), dtype=np.uint8)
    out_layers[:, :, 3] = layer_channel
    out_composite = src_img.copy()
    out_background = np.ones((512, 512, 3), dtype=np.uint8) * 255
    out_img = {}
    out_img["background"] = out_background
    out_img["layers"] = [out_layers]
    out_img["composite"] = out_composite
    return out_img


# 内部提取线稿构建展示格式
def build_image_editor_input(src_img):
    src_img = normalize_editor_image(src_img, target_size=512)
    layer_channel = np.where(src_img != 255, 255 - src_img, 0)
    layer_channel = layer_channel[:, :, 0]
    out_layers = np.zeros((512, 512, 4), dtype=np.uint8)
    out_layers[:, :, 3] = layer_channel
    out_composite = src_img.copy()
    out_background = np.ones((512, 512, 3), dtype=np.uint8) * 255
    out_img = {}
    out_img["background"] = out_background
    out_img["layers"] = [out_layers]
    out_img["composite"] = out_composite
    return out_img


# 展示例子
def handle_example(sketch, appearance):
    out_sketch = build_image_editor_output(sketch)
    return out_sketch, appearance


def load_generation_example(sketch_path, appearance_path):
    sketch_img = Image.open(sketch_path) if isinstance(sketch_path, str) else sketch_path
    appearance_img = Image.open(appearance_path) if isinstance(appearance_path, str) else appearance_path
    out_sketch = build_image_editor_output(sketch_img)
    out_appearance = handle_upload_appearance(appearance_img)
    return out_sketch, out_appearance


def _resolve_checkpoint_path(path):
    if path is None or os.path.isabs(path):
        resolved = path
    else:
        resolved = os.path.join(_base_dir, path)
    default_checkpoint = os.path.normpath("checkpoints/model.pkl")
    legacy_checkpoints = [
        os.path.join(_base_dir, "output", "model.pkl"),
        os.path.join(_base_dir, "output", "new_jittor_remap_v2.pkl"),
        os.path.join(_base_dir, "output", "new_jittor_remap.pkl"),
    ]
    if path is not None and os.path.normpath(path) == default_checkpoint and not os.path.exists(resolved):
        for legacy_checkpoint in legacy_checkpoints:
            if os.path.exists(legacy_checkpoint):
                return legacy_checkpoint
    return resolved


import argparse as _argparse
_ap = _argparse.ArgumentParser(description="SketchFaceGS Jittor Gradio App")
_ap.add_argument("--checkpoint", default="checkpoints/model.pkl",
                 help="Path to model checkpoint (.pkl)")
_ap.add_argument("--config", default="configs/train.yaml",
                 help="Path to config YAML")
_ap.add_argument("--port", type=int, default=7860)
_ap.add_argument("--share", action="store_true")
_args, _ = _ap.parse_known_args()

CHECKPOINT_PATH = _resolve_checkpoint_path(_args.checkpoint)
CONFIG_PATH = _args.config
pipeline = AISketchAndColorTo3D(CHECKPOINT_PATH, CONFIG_PATH)

# 1. Model 1 专用：极速预览（仅返回右侧 ImageSlider）
def on_angle_preview_right_only(alpha, beta):
    res = pipeline.render_from_cached_model(
        -alpha, beta, 2.7, return_sketch=False, gen_sketch=False
    )
    rgb_img = res[0]

    # 2. 暴力降分辨率 (ImageSlider 右侧)
    preview_size = (512, 512)
    rgb_show = rgb_img.resize(preview_size, resample=Image.NEAREST)

    # 3. 准备右侧占位图
    empty_right = Image.new("RGB", preview_size, (30, 30, 30))

    # 4. 【核心修改】保存为临时 JPEG 文件，获取路径
    # 这一步依然保留了 JPEG 的小体积优势，但返回的是短路径，不会报错
    slider_left_path = save_as_jpeg(rgb_show, quality=40)
    slider_right_path = save_as_jpeg(empty_right, quality=20)

    return [slider_left_path, slider_right_path]

# 2. Model 1 专用：全量高质量渲染（仅更新右侧）
def on_angle_gen_high_quality(alpha, beta):
    
    # 高质量渲染 + 线稿提取
    # 这里的 return_sketch=True 会调用 pipeline 内部逻辑
    res = pipeline.render_from_cached_model(
        -alpha, beta, 2.7, return_sketch=False, gen_sketch=False, mode="edit"
    )

    
    return res

# 3. Model 1 专用：生成并重置滑块
def on_model1_generate_with_reset(sketch_pack, color_ref, alpha, beta):
    # 执行生成
    # 注意：这里我们传入 alpha/beta 是为了让模型在当前视角生成，
    # 但生成完后，我们要把滑块值归零。
    res = pipeline.generate_with_input_generate_gaussian_model(
        sketch_pack, color_ref, -alpha, beta #beta - 12.0
    )
    # 返回：[渲染结果, 重置Alpha为0, 重置Beta为0]
    return res, 0, 0
# 生成
def on_generate(sketch_pack, color_ref, alpha, beta):
    return pipeline.generate_with_input_generate_gaussian_model(
        sketch_pack, color_ref, alpha=-alpha, beta=beta#beta - 12.0
    )

def on_model3_generate_with_reset(seed_input, rt_active):
    """
    包装函数：执行生成，并返回重置后的滑块数值 + 恢复定时器 + 新版本号
    """
    # 1. 确保 pipeline 内部状态也重置 (对应 UI 的默认值)
    pipeline.opacity = 0.6 
    # 注意：alpha 和 beta 在 edit_without_input_generate_gaussian_model 内部已经调用 self.reset_cam() 重置了
    
    # 2. 执行核心生成逻辑（内部会递增 _state_version）
    # 返回值结构: [[render, ellipsoid], sketch_dict, seed]
    pipeline._begin_action(blocking=True)
    try:
        results = pipeline.edit_without_input_generate_gaussian_model(seed_input)
    finally:
        pipeline._end_action()
    
    # 3. 返回给 UI 的所有数据 + 定时器恢复 + 新版本号
    return (
        results[0],  # mix_output (ImageSlider)
        results[1],  # sketch_upload (ImageEditor)
        results[2],  # seed_current (Textbox)
        0,           # alpha_slider (重置为 0)
        0,           # beta_slider (重置为 0)
        0.7,         # sketch_opacity_slider (重置为 0.7)
        gr.update(active=rt_active),  # real_time_edit (恢复定时器)
        pipeline._state_version,  # state_version (同步版本号到客户端)
    )
# 编辑
def on_edit_no_sketch(sketch_edit, alpha, beta, version=None):
    print("edit not return sketch")
    blocking = version is None
    if not pipeline._begin_action(blocking=blocking):
        print("[Skip] on_edit_no_sketch busy; returning cached output.")
        return _cached_mix_output(-alpha, beta)
    try:
        return pipeline.edit_gaussian_model(
            sketch_edit,
            -alpha,
            beta,
            return_sketch=False,
            gen_sketch=False,
            keep_sketch=True,
            state_version=version,
        )
    finally:
        pipeline._end_action()

# 不透明度变化中（合并停止定时器 + 更新，避免队列交错）
def on_opacity_change_with_stop(opacity):
    pipeline._state_version += 1
    # 只更新内部状态，不返回 sketch_upload 更新
    # 避免 ImageEditor DOM 重渲染打断滑条拖拽；松手后由 on_opacity_release 统一刷新
    pipeline.opacity = opacity
    return gr.update(), gr.update(active=False)


def on_opacity_change_with_stop_m3(opacity):
    if pipeline.active_tab != "model3":
        return gr.update(), gr.update(active=False)
    return on_opacity_change_with_stop(opacity)


def on_opacity_release(opacity, rt_active):
    """
    松手时：重刷最终 opacity（清除快速拖动脏状态），更新 last_sketch 基准，恢复定时器。
    下次 tick mask 为空自动跳过，不触发前馈。
    """
    sketch_update = pipeline.update_sketch_opacity(opacity)
    pipeline._state_version += 1
    return sketch_update, gr.update(active=rt_active), pipeline._state_version


def on_opacity_release_m3(opacity, rt_active):
    if pipeline.active_tab != "model3":
        return gr.update(), gr.update(active=False), pipeline._state_version
    return on_opacity_release(opacity, rt_active)

def _no_op_preview():
    """当 active_tab 不匹配时，静默返回空更新，不渲染"""
    return gr.update(), gr.update(), gr.update(active=False)


def _cached_preview_dual(alpha, beta, opacity):
    rendered = getattr(pipeline, "last_rendered_image", None)
    if rendered is None:
        rendered = create_default_512_image()
    bg_image = getattr(pipeline, "last_rendered_image_back", None)
    if bg_image is None:
        bg_image = rendered
    preview_size = (512, 512)
    rgb_show = rendered.resize(preview_size, resample=Image.NEAREST)
    ellipsoid_image = getattr(pipeline, "last_ellipsoid_image", None)
    if ellipsoid_image is None:
        ellipsoid_np = pipeline.gl_worker.sync_render(-alpha, beta, 2.7)
        ellipsoid_image = Image.fromarray(ellipsoid_np)
    slider_left_path = save_as_jpeg(rgb_show, quality=40)
    slider_right_path = save_as_jpeg(
        ellipsoid_image.resize(preview_size, resample=Image.NEAREST), quality=20
    )
    left_editor_data = build_sketch_on_rgb_output(
        bg_image, None, ratio=opacity, preview_mode=True, target_size=512
    )
    return [slider_left_path, slider_right_path], left_editor_data


def _cached_full_sync(alpha, beta, opacity):
    rendered = getattr(pipeline, "last_rendered_image", None)
    if rendered is None:
        rendered = create_default_512_image()
    bg_image = getattr(pipeline, "last_rendered_image_back", None)
    if bg_image is None:
        bg_image = rendered
    ellipsoid_image = getattr(pipeline, "last_ellipsoid_image", None)
    if ellipsoid_image is None:
        ellipsoid_np = pipeline.gl_worker.sync_render(-alpha, beta, 2.7)
        ellipsoid_image = Image.fromarray(ellipsoid_np)
    current_sketch = getattr(pipeline, "display_sketch", None)
    if current_sketch is None:
        current_sketch = getattr(pipeline, "last_sketch", None)
    final_left_data = build_sketch_on_rgb_output(
        bg_image, current_sketch, ratio=opacity
    )
    return [rendered, ellipsoid_image], final_left_data


def _cached_mix_output(alpha, beta):
    rendered = getattr(pipeline, "last_rendered_image", None)
    if rendered is None:
        rendered = create_default_512_image()
    ellipsoid_image = getattr(pipeline, "last_ellipsoid_image", None)
    if ellipsoid_image is None:
        ellipsoid_np = pipeline.gl_worker.sync_render(alpha, beta, 2.7)
        ellipsoid_image = Image.fromarray(ellipsoid_np)
    return rendered, ellipsoid_image


# 角度变化中（合并停止定时器 + 预览，避免队列阻塞）
def on_angle_preview_dual_with_stop(alpha, beta, opacity):
    """合并 stop_real_time + 预览，单次回调输出全部组件"""
    pipeline._state_version += 1
    timer_update = gr.update(active=False)
    preview = on_angle_preview_dual(alpha, beta, opacity)
    return preview[0], preview[1], timer_update


def on_angle_preview_dual_with_stop_m3(alpha, beta, opacity):
    if pipeline.active_tab != "model3":
        return _no_op_preview()
    return on_angle_preview_dual_with_stop(alpha, beta, opacity)


def on_angle_full_sync_m3(alpha, beta, opacity):
    if pipeline.active_tab != "model3":
        return gr.update(), gr.update()
    return on_angle_full_sync(alpha, beta, opacity)

# 角度变化中
def on_angle_preview_dual(alpha, beta, opacity):
    if not pipeline._begin_action(blocking=False):
        print("[Skip] Rotation preview busy; returning cached preview.")
        return _cached_preview_dual(alpha, beta, opacity)
    try:
        pipeline.alpha = -alpha
        pipeline.beta = beta
        pipeline.radius = 2.7

        if pipeline._rotation_should_use_cached():
            print("[Skip] Rotation preview while edit is stabilizing.")
            return _cached_preview_dual(alpha, beta, opacity)

        main_np, bg_np = pipeline._render_both_np(-alpha, beta, 2.7, c=pipeline.c, allow_drop=True)

        ellipsoid_image = pipeline.gl_worker.sync_render(-alpha, beta, 2.7)
        pipeline.last_ellipsoid_image = Image.fromarray(ellipsoid_image)

        rendered = Image.fromarray(main_np)
        pipeline.last_rendered_image = rendered

        preview_size = (512, 512)
        rgb_show = rendered.resize(preview_size, resample=Image.NEAREST)
        empty_right = Image.new("RGB", preview_size, (30, 30, 30))
        slider_left_path = save_as_jpeg(rgb_show, quality=40)
        slider_right_path = save_as_jpeg(empty_right, quality=20)

        bg_image = Image.fromarray(bg_np)
        left_editor_data = build_sketch_on_rgb_output(
            bg_image, None, ratio=opacity, preview_mode=True, target_size=512
        )

        return [slider_left_path, slider_right_path], left_editor_data
    finally:
        pipeline._end_action()


# 2. 松手后的全量同步：提取线稿，恢复展示
def on_angle_full_sync(alpha, beta, opacity):
    pipeline._begin_action(blocking=True)
    try:
        pipeline.alpha = -alpha
        pipeline.beta = beta
        pipeline.radius = 2.7

        if pipeline._rotation_should_use_cached():
            print("[Skip] Rotation full sync while edit is stabilizing.")
            return _cached_full_sync(alpha, beta, opacity)

        res_list = pipeline.render_from_cached_model(
            -alpha, beta, 2.7, return_sketch=True, gen_sketch=True, mode="edit"
        )

        final_left_data = build_sketch_on_rgb_output(
            pipeline.last_rendered_image_back, pipeline.last_sketch, ratio=opacity
        )
        return res_list[0], final_left_data
    finally:
        pipeline._end_action()


# 编辑的角度变化
def on_angle_edit(alpha, beta):
    pipeline.c = None
    time1 = time.time()
    res = pipeline.render_from_cached_model(
        -alpha, beta, 2.7, return_sketch=True, mode="edit"
    )
    time2 = time.time()
    return res


# 生成的角度变化
def on_angle_gen(alpha, beta):
    pipeline.c = None
    print(f"beta {beta}")
    time1 = time.time()
    res = pipeline.render_from_cached_model(-alpha, beta , 2.7, gen_sketch=False)#beta - 12.0
    time2 = time.time()
    return res


# 改变半径
# def on_angle_with_radius(alpha, beta, radius):
#     pipeline.c = None
#     return pipeline.render_from_cached_model(-alpha, beta , radius, return_sketch=True)


# 上传线稿
def on_upload(sketch_pack):
    return build_image_editor_output(sketch_pack)


def cache_current_editor_sketch(sketch_pack):
    if sketch_pack is None or not isinstance(sketch_pack, dict):
        return

    composite = sketch_pack.get("composite")
    if composite is None or pipeline.last_rendered_image_back is None:
        return

    if not isinstance(composite, np.ndarray):
        composite = np.array(composite)

    background = np.array(pipeline.last_rendered_image_back)
    extracted = extract_sketch_from_composite(composite, background)
    if extracted is not None:
        pipeline.display_sketch = extracted


# 实时开启
def start_real_time():
    print(f"start_real_time")
    return gr.update(active=True)


# 关闭实时
def stop_real_time():
    print(f"stop_real_time")
    pipeline._state_version += 1  # 使所有已排队的旧定时器事件立即失效
    return gr.update(active=False)


def set_real_time_state_and_sync(rt_active):
    pipeline._state_version += 1
    print("start_real_time" if rt_active else "stop_real_time")
    return rt_active, gr.update(active=rt_active), pipeline._state_version


# 白色初始图片
def create_default_512_image():
    """创建一张默认的512×512图片（白色背景）"""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    return Image.fromarray(img)


# 辅助函数：决定是否重启 Timer
def resume_timer_if_needed(rt_active):
    return gr.update(active=rt_active)

def resume_timer_and_sync_version(rt_active):
    """恢复定时器 + 递增版本号，防止旋转前的旧定时器事件污染"""
    pipeline._state_version += 1
    return gr.update(active=rt_active), pipeline._state_version


def save_last_rendered_image_back(picture):
    """将 pipeline.last_rendered_image_back 按时间命名保存到 test_output 目录。"""
    # if not hasattr(pipeline, "last_rendered_image_back") or pipeline.last_rendered_image_back is None:
    #     return None

    output_dir = os.path.join(os.getcwd(), "test_output")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = os.path.join(output_dir, f"render_back_{timestamp}.png")

    picture.save(file_path)
    # return file_path


# 对齐框
custom_css = """
.image-slider-wrapper {
    border-style: solid;
    border-color: var(--block-background-fill);
    border-width: 25px 8px 50px 8px;  /* 上 右 下 左 */
    border-radius: var(--radius-lg);
    background-color: var(--block-background-fill);
    padding: var(--spacing-lg);
    box-shadow: var(--block-shadow);
    display: inline-block;
    height: 610px;   /* ✅ 强制和中间对齐 */
    box-sizing: border-box;
}

/* ===== 修复 Gradio ImageEditor 在 Retina 屏幕(iPad)上画布偏小的 bug =====
   原因：pixi.ts 中 maxWidth = width/devicePixelRatio，iPad ratio=2 导致画布只有一半大
   [data-testid="image"] 是 ImageEditor 内部 .image-container 的属性 */
[data-testid="image"] canvas {
    max-width: 100% !important;
    max-height: 100% !important;
}
"""

alpha_max = 40
alpha_min = -40
beta_max = 30
beta_min = -30
slider_len = 450


def model_1():
    with gr.Blocks(theme=gr.themes.Soft(), delete_cache=(60, 120)) as m1:
        # [新增] 状态机：记录用户是否手动开启了实时模式（默认为 False）
        rt_status = gr.State(False)
        with gr.Row(equal_height=False, elem_classes="three-col-row"):
            with gr.Column(min_width=270, scale=0):
                gr.Markdown("### Upload Image")
                color_ref_upload = gr.Image(
                    type="pil",
                    label=None,
                    format="png",
                )
                generate_button = gr.Button("🚀 Generate 3D Model", variant="primary")
                enable_rt_button = gr.Button("Enable Realtime")
                disable_rt_button = gr.Button("Disable realtime")

            with gr.Column(min_width=600, scale=1):
                gr.Markdown("### Upload Sketch")
                sketch_upload = gr.ImageEditor(
                    value=build_image_editor_output(create_default_512_image()),
                    type="pil",
                    format="png",
                    label=None,
                    layers=False,
                    height=610,
                    width=600,
                    brush=gr.Brush(2, [0], [0], "fixed"),
                    interactive=True,
                )

            with gr.Column(scale=1, min_width=540):
                gr.Markdown("### View")
                with gr.Group(elem_classes=["image-slider-wrapper"]):
                    mix_output = ImageSlider(
                        label="Render",
                        type="pil",
                        height=512,
                        width=512,
                    )

        gr.Markdown("### Camera Control")
        with gr.Row(equal_height=True, elem_classes="three-col-row"):
            alpha_slider = gr.Slider(
                alpha_min,
                alpha_max,
                0,
                1,
                label="Alpha",
                min_width=slider_len,
            )
            alpha_reset = gr.Button("Reset Alpha", min_width=75)
            beta_slider = gr.Slider(
                beta_min, beta_max, 0, 1, label="Beta", min_width=slider_len
            )
            beta_reset = gr.Button("Reset Beta", min_width=75)

        gr.Markdown("### Examples")
        example_sketch_input = gr.Image(type="filepath", visible=False)
        example_color_input = gr.Image(type="filepath", visible=False)
        gr.Examples(
            examples=[
                
                ["./examples/sketch_1.png", "./examples/color_1.png"],
                ["./examples/sketch_2.png", "./examples/color_2.png"],
               
               
            ],
            inputs=[example_sketch_input, example_color_input],
            outputs=[sketch_upload, color_ref_upload],
            fn=load_generation_example,
            label="Examples",
            run_on_click=True,
            cache_examples=False,
        )

        real_time_edit = gr.Timer(0.8, active=False)
        

        # 1. [修改] 实时模式开关：现在会记忆 rt_status
        enable_rt_button.click(
            fn=lambda: (True, gr.update(active=True)), 
            outputs=[rt_status, real_time_edit]
        )
        disable_rt_button.click(
            fn=lambda: (False, gr.update(active=False)), 
            outputs=[rt_status, real_time_edit]
        )


        # 2. [修改] 生成按钮：增加复位逻辑 + 尊重 RT 状态
        generate_button.click(
            fn=stop_real_time, outputs=real_time_edit # 先强制停，防止生成过程中冲突
        ).then(
            fn=on_model1_generate_with_reset,
            inputs=[sketch_upload, color_ref_upload, alpha_slider, beta_slider],
            outputs=[mix_output, alpha_slider, beta_slider], # [新增] alpha/beta 复位到 0
        ).then(
            # [新增] 生成结束后，根据之前的状态决定是否重启实时
            fn=resume_timer_if_needed, inputs=[rt_status], outputs=real_time_edit
        )

        sketch_upload.upload(
            fn=stop_real_time, inputs=None, outputs=real_time_edit
        ).then(on_upload, inputs=sketch_upload, outputs=sketch_upload)

       
        
        color_ref_upload.upload(
            fn=stop_real_time, inputs=None, outputs=real_time_edit
        ).then(
            handle_upload_appearance, inputs=color_ref_upload, outputs=color_ref_upload
        )

        for slider in [alpha_slider, beta_slider]:
            # 拖动中：暂停 Timer，极速更新右侧，不转圈
            slider.change(
                fn=stop_real_time, outputs=real_time_edit, show_progress="hidden"
            ).then(
                fn=on_angle_preview_right_only,
                inputs=[alpha_slider, beta_slider],
                outputs=[mix_output],
                show_progress="hidden", queue=False # [优化] 不排队，直接丢弃过时请求
            )
            
            # 松手时：高质量渲染 + 根据 rt_status 决定是否恢复 Timer
            slider.release(
                fn=on_angle_gen_high_quality, # 高清版
                inputs=[alpha_slider, beta_slider],
                outputs=[mix_output],
                show_progress="hidden"
            ).then(
                fn=resume_timer_if_needed, inputs=[rt_status], outputs=real_time_edit
            )

        # 4. [修改] 重置按钮：重置后同样需要恢复 Timer 状态
        alpha_reset.click(
            fn=stop_real_time, outputs=real_time_edit
        ).then(lambda: 0, outputs=[alpha_slider]).then(
            fn=on_angle_gen_high_quality, inputs=[alpha_slider, beta_slider], outputs=[mix_output]
        ).then(fn=resume_timer_if_needed, inputs=[rt_status], outputs=real_time_edit)

        beta_reset.click(
            fn=stop_real_time, outputs=real_time_edit
        ).then(lambda: 0, outputs=[beta_slider]).then(
            fn=on_angle_gen_high_quality, inputs=[alpha_slider, beta_slider], outputs=[mix_output]
        ).then(fn=resume_timer_if_needed, inputs=[rt_status], outputs=real_time_edit)

        # 5. [优化] 定时器触发：增加隐藏进度条
        real_time_edit.tick(
            fn=on_generate,
            inputs=[sketch_upload, color_ref_upload, alpha_slider, beta_slider],
            outputs=mix_output,
            show_progress="hidden"
        )
    return m1, real_time_edit
        
def model_2():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        delete_cache=(60, 120),
    ):
        with gr.Row(equal_height=False, elem_classes="three-col-row"):
            with gr.Column(min_width=270, scale=0):
                gr.Markdown("### Upload Image")
                color_ref_upload = gr.Image(
                    type="pil",
                    format="png",
                    image_mode="RGB",
                )
                generate_button = gr.Button(
                    "Step1: 🚀 Generate 3D model", variant="primary"
                )
                edit_button = gr.Button("Step2: 🚀 Edit 3D model", variant="primary")
                enable_rt_button = gr.Button("Enable Realtime")
                disable_rt_button = gr.Button("Disable Realtime")
            with gr.Column(min_width=600, scale=0):
                gr.Markdown("### Upload Sketch")
                sketch_upload = gr.ImageEditor(
                    value=create_default_512_image(),  # 确保返回RGB模式的numpy数组或PIL Image
                    format="png",
                    type="pil",
                    layers=False,
                    height=610,
                    width=600,
                    brush=gr.Brush(2, [0], [0], "fixed"),
                    interactive=True,
                )

            with gr.Column(min_width=540, scale=1):
                gr.Markdown("### View")
                with gr.Group(elem_classes=["image-slider-wrapper"]):
                    mix_output = ImageSlider(
                        label="Render",
                        type="pil",
                        height=512,
                        width=512,
                    )

        gr.Markdown("### Camera Control")
        with gr.Row(equal_height=True, elem_classes="three-col-row"):
            alpha_slider = gr.Slider(
                alpha_min,
                alpha_max,
                0,
                1,
                label="Alpha",
                min_width=slider_len,
            )
            alpha_reset = gr.Button("重置", min_width=75)
            beta_slider = gr.Slider(
                beta_min, beta_max, 0, 1, label="Beta", min_width=slider_len
            )
            beta_reset = gr.Button("重置", min_width=75)

        gr.Markdown("### Examples")
        gr.Examples(
            examples=[
                ["./examples/sketch.jpg", "./examples/color.jpg"],
                ["./examples/sketch_1.png", "./examples/color.jpg"],
                ["./examples/sketch_2.jpg", "./examples/color.jpg"],
            ],
            inputs=[sketch_upload, color_ref_upload],
            outputs=[sketch_upload, color_ref_upload],
            fn=handle_example,
            label="Examples",
            run_on_click=True,
            cache_examples=False,
        )

        real_time_edit = gr.Timer(0.8, active=False)
        real_time_edit.tick(
            fn=on_edit_no_sketch,
            inputs=[sketch_upload, alpha_slider, beta_slider],
            outputs=mix_output,
        )

        generate_button.click(
            fn=stop_real_time, inputs=None, outputs=real_time_edit
        ).then(
            fn=pipeline.edit_with_input_generate_gaussian_model,
            inputs=[sketch_upload, color_ref_upload],
            outputs=[mix_output, sketch_upload],
        )

        edit_button.click(fn=stop_real_time, inputs=None, outputs=real_time_edit).then(
            fn=on_edit_no_sketch,
            inputs=[sketch_upload, alpha_slider, beta_slider],
            outputs=mix_output,
        )

        sketch_upload.upload(on_upload, inputs=sketch_upload, outputs=sketch_upload)
        color_ref_upload.upload(
            handle_upload_appearance, inputs=color_ref_upload, outputs=color_ref_upload
        )
        alpha_slider.input(fn=stop_real_time, inputs=None, outputs=real_time_edit).then(
            fn=on_angle_edit,
            inputs=[alpha_slider, beta_slider],
            outputs=[mix_output, sketch_upload],
            queue=False,
            show_progress=False,
        ).success(fn=start_real_time, inputs=None, outputs=real_time_edit)

        beta_slider.input(fn=stop_real_time, inputs=None, outputs=real_time_edit).then(
            fn=on_angle_edit,
            inputs=[alpha_slider, beta_slider],
            outputs=[mix_output, sketch_upload],
            queue=False,
            show_progress=False,
        ).success(fn=start_real_time, inputs=None, outputs=real_time_edit)

        alpha_reset.click(fn=stop_real_time, inputs=None, outputs=real_time_edit).then(
            lambda: 0, outputs=[alpha_slider]
        ).success(fn=start_real_time, inputs=None, outputs=real_time_edit)
        beta_reset.click(fn=stop_real_time, inputs=None, outputs=real_time_edit).then(
            lambda: 0, outputs=[beta_slider]
        ).success(fn=start_real_time, inputs=None, outputs=real_time_edit)
        enable_rt_button.click(fn=start_real_time, inputs=None, outputs=real_time_edit)
        disable_rt_button.click(fn=stop_real_time, inputs=None, outputs=real_time_edit)


def model_3():

    with gr.Blocks(theme=gr.themes.Soft(), delete_cache=(60, 120)) as m3:
        rt_status = gr.State(False)
        with gr.Row(equal_height=False, elem_classes="three-col-row"):
            with gr.Column(scale=0, min_width=600):
                gr.Markdown("### Generate Sketch")
                sketch_upload = gr.ImageEditor(
                    value=create_default_512_image(),
                    type="pil",
                    format="png",
                    layers=False,
                    height=610,
                    width=600,
                    # 初始笔刷颜色带透明度
                    brush=gr.Brush(2, colors=["#08080899"], color_mode="fixed"),
                )
                with gr.Row():
                    generate_button = gr.Button(
                        "Step1: 🚀 Generate 3D Model", variant="primary"
                    )
                    edit_button = gr.Button(
                        "Step2: 🚀 Edit 3D Model", variant="primary"
                    )

            with gr.Column(scale=0, min_width=540):
                gr.Markdown("### View")
                with gr.Group(elem_classes=["image-slider-wrapper"]):
                    mix_output = ImageSlider(
                        label="Render", type="pil", height=512, width=512
                    )
                with gr.Row():
                    enable_rt_button = gr.Button("Enable Realtime")
                    disable_rt_button = gr.Button("Disable Realtime")

        gr.Markdown("### Control")
        with gr.Row():
            alpha_slider = gr.Slider(
                -40, 40, 0, step=1, label="Alpha", min_width=slider_len
            )
            beta_slider = gr.Slider(
                -30, 30, 0, step=1, label="Beta", min_width=slider_len
            )

        with gr.Row():
            seed_current = gr.Textbox(label="Current Seed")
            seed_input = gr.Textbox(label="Input Seed (Optional)")
            sketch_opacity_slider = gr.Slider(
                0.0, 1.0, 0.7, step=0.05, label="Sketch Opacity"
            )

        # --- 核心修复：添加按钮点击事件 ---
        # 状态版本号：每次生成/旋转完成后递增，定时器事件携带触发时的版本号用于同步
        state_version = gr.State(0)

        # 定时器逻辑
        real_time_edit = gr.Timer(0.8, active=False)
        tick_event = real_time_edit.tick(
            fn=on_edit_no_sketch,
            inputs=[sketch_upload, alpha_slider, beta_slider, state_version],
            outputs=[mix_output],
            show_progress="hidden",
        )

        # A. 实时模式开关：同时更新 Timer 和 rt_status 状态
        enable_rt_button.click(
            fn=lambda: True,
            outputs=[rt_status],
        ).then(
            fn=resume_timer_and_sync_version,
            inputs=[rt_status],
            outputs=[real_time_edit, state_version],
        )
        disable_rt_button.click(
            fn=lambda: False,
            outputs=[rt_status],
        ).then(
            fn=set_real_time_state_and_sync,
            inputs=[rt_status],
            outputs=[rt_status, real_time_edit, state_version],
        )
        
        # 1. 生成按钮逻辑 (Step 1)
        generate_button.click(
            fn=stop_real_time, outputs=real_time_edit,  # 真正停止定时器
        ).then(
            # 改用新的包装函数（同时恢复定时器，避免额外 .then 造成卡顿）
            fn=on_model3_generate_with_reset, 
            inputs=[seed_input, rt_status],
            outputs=[
                mix_output, 
                sketch_upload, 
                seed_current, 
                alpha_slider,          # 目标：Alpha
                beta_slider,           # 目标：Beta
                sketch_opacity_slider, # 目标：Opacity
                real_time_edit,        # 定时器恢复
                state_version,         # 版本号同步到客户端
            ],
        )

        # 2. 编辑按钮逻辑 (Step 2)
        edit_button.click(fn=stop_real_time, outputs=real_time_edit).then(
            fn=on_edit_no_sketch,
            inputs=[sketch_upload, alpha_slider, beta_slider],
            outputs=[mix_output],
            show_progress="hidden",
        ).then(
            fn=resume_timer_and_sync_version,
            inputs=[rt_status],
            outputs=[real_time_edit, state_version],
        )
        # --- 旋转与滑块逻辑 ---

        # 拖动中：极速预览（model3 专属，queue=False 丢弃堆积帧，避免 GPU 队列积压）
        alpha_slider.input(
            fn=on_angle_preview_dual_with_stop_m3,
            inputs=[alpha_slider, beta_slider, sketch_opacity_slider],
            outputs=[mix_output, sketch_upload, real_time_edit],
            show_progress="hidden",
            queue=False,
        )
        beta_slider.input(
            fn=on_angle_preview_dual_with_stop_m3,
            inputs=[alpha_slider, beta_slider, sketch_opacity_slider],
            outputs=[mix_output, sketch_upload, real_time_edit],
            show_progress="hidden",
            queue=False,
        )

        # 松手时：全量高质量同步（model3 专属）
        alpha_slider.release(
            fn=on_angle_full_sync_m3,
            inputs=[alpha_slider, beta_slider, sketch_opacity_slider],
            outputs=[mix_output, sketch_upload],
            show_progress="hidden",
        ).then(fn=resume_timer_and_sync_version, inputs=[rt_status], outputs=[real_time_edit, state_version])
        beta_slider.release(
            fn=on_angle_full_sync_m3,
            inputs=[alpha_slider, beta_slider, sketch_opacity_slider],
            outputs=[mix_output, sketch_upload],
            show_progress="hidden",
        ).then(fn=resume_timer_and_sync_version, inputs=[rt_status], outputs=[real_time_edit, state_version])

        # 透明度滑块：拖动停止 + 更新（model3 专属），松手重刷最终值并恢复定时器
        sketch_opacity_slider.input(
            fn=on_opacity_change_with_stop_m3,
            inputs=[sketch_opacity_slider],
            outputs=[sketch_upload, real_time_edit],
            show_progress="hidden",
            queue=False,
        )
        sketch_opacity_slider.release(
            fn=on_opacity_release_m3,
            inputs=[sketch_opacity_slider, rt_status],
            outputs=[sketch_upload, real_time_edit, state_version],
            queue=False,
        )
        
        with gr.Row():
            update_bg_btn = gr.Button("🔄 Set Current as Background", variant="secondary")

        update_bg_btn.click(
            fn=stop_real_time, outputs=real_time_edit # 1. 暂停实时，防止冲突
        ).then(
            fn=pipeline.update_current_to_background, # 2. 执行更新逻辑
            inputs=None,
            outputs=[sketch_upload],                  # 3. 更新线稿编辑器背景
        ).then(
            fn=resume_timer_and_sync_version, inputs=[rt_status], outputs=[real_time_edit, state_version] # 4. 恢复实时状态
        )

        sketch_upload.input(fn=cache_current_editor_sketch, inputs=[sketch_upload], outputs=None, queue=False)
        
    return m3, real_time_edit  # 确保返回 Blocks 实例


def main():
    _head_html = '<meta name="viewport" content="width=1600, user-scalable=yes">'

    _resize_js = """
    () => {
        function fireResize() {
            window.dispatchEvent(new Event('resize'));
        }
        setTimeout(fireResize, 500);
        setTimeout(fireResize, 1500);
        setTimeout(fireResize, 3000);
        document.addEventListener('click', (e) => {
            if (e.target.closest('.tab-nav button')) {
                setTimeout(fireResize, 300);
                setTimeout(fireResize, 800);
            }
        });
    }
    """
    with gr.Blocks(title="SketchFaceGaussian", css=custom_css, head=_head_html, js=_resize_js) as demo:
        gr.Markdown("# 🎨 SketchFaceGS (Jittor)")
        
        with gr.Tabs() as tabs:
            with gr.Tab("Generation Mode", id="t1") as t1_ui:
                model_1() 
                
            with gr.Tab("Edit Model: Generate", id="t3") as t3_ui:
                m3_blocks, timer_m3 = model_3()

        # Tab 切换守卫：切到 t1 时停 t3 的 timer
        def kill_all_timers():
            print("Switching tabs: Killing all timers...")
            return gr.update(active=False)

        t1_ui.select(
            fn=kill_all_timers,
            inputs=None,
            outputs=[timer_m3]
        )

    try:
        try:
            demo.launch(server_name="0.0.0.0", server_port=_args.port, share=_args.share)
        except ValueError as e:
            if "localhost is not accessible" not in str(e) or _args.share:
                raise
            demo.launch(server_name="0.0.0.0", server_port=_args.port, share=True)
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()