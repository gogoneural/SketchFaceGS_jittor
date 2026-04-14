import torch
import numpy as np
import os
import argparse
from torchvision.ops import roi_align
from tqdm import tqdm

# --- 1. 边界框计算辅助函数 (保持不变) ---

def get_bounding_box(points, scale=1.0):
    """对称放大边界框"""
    x_coords, y_coords = points[:, 0], points[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    if scale != 1.0:
        width, height = x_max - x_min, y_max - y_min
        center_x, center_y = x_min + width / 2, y_min + height / 2
        width, height = width * scale, height * scale
        x_min, x_max = center_x - width / 2, center_x + width / 2
        y_min, y_max = center_y - height / 2, center_y + height / 2

    return int(x_min), int(y_min), int(x_max), int(y_max)

def get_eyebrow_bounding_box(points, height_scale=1.0, width_scale=1.0):
    """眉毛：上边界不动，向下拉伸"""
    x_coords, y_coords = points[:, 0], points[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    width = x_max - x_min
    center_x = x_min + width / 2
    new_width = width * width_scale
    x_min, x_max = center_x - new_width / 2, center_x + new_width / 2

    height = y_max - y_min
    new_height = height * height_scale
    y_max = y_min + new_height

    return int(x_min), int(y_min), int(x_max), int(y_max)

def get_nose_bounding_box(points, width_scale=1.0):
    """鼻子：上下不动，左右拉伸"""
    x_coords, y_coords = points[:, 0], points[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    width = x_max - x_min
    center_x = x_min + width / 2
    new_width = width * width_scale
    x_min = center_x - new_width / 2
    x_max = center_x + new_width / 2

    return int(x_min), int(y_min), int(x_max), int(y_max)

def get_eye_bounding_box(points, width_scale=1.0, height_scale=1.0):
    """眼睛：独立缩放宽高"""
    x_coords, y_coords = points[:, 0], points[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    width, height = x_max - x_min, y_max - y_min
    center_x, center_y = x_min + width / 2, y_min + height / 2
    
    new_width, new_height = width * width_scale, height * height_scale
    
    x_min, x_max = center_x - new_width / 2, center_x + new_width / 2
    y_min, y_max = center_y - new_height / 2, center_y + new_height / 2

    return int(x_min), int(y_min), int(x_max), int(y_max)


# --- 2. 主集成函数 ---

def extract_facial_components_with_rois(image_batch: torch.Tensor, fa_model, output_size: int = 120):
    """
    修改版：除了返回裁剪的组件，还返回用于裁剪的RoIs。
    """
    B, C, H, W = image_batch.shape
    
    # --- 各种参数和索引定义 (和之前版本一样) ---
    eye_width_scale, eye_height_scale = 1.6, 1.75
    nose_width_scale = 1.6
    mouth_scale = 1.2
    eyebrow_height_scale, eyebrow_width_scale = 1.8, 1.2
    indices = {
        "left_eyebrow": list(range(17, 22)), "right_eyebrow": list(range(22, 27)),
        "nose": list(range(27, 36)),
        "left_eye": list(range(36, 42)), "right_eye": list(range(42, 48)),
        "mouth": list(range(48, 68))
    }
    
    rois = {key: [] for key in indices.keys()}

    for i in range(B):
        img_np = (image_batch[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        preds = fa_model.get_landmarks(img_np)
        
        if not preds:
            for key in rois.keys():
                rois[key].append(torch.tensor([i, 0, 0, 0, 0]))
            continue

        landmarks = preds[0]
        points = {key: landmarks[idx] for key, idx in indices.items()}
        bboxes = {
            "left_eyebrow": get_eyebrow_bounding_box(points["left_eyebrow"], eyebrow_height_scale, eyebrow_width_scale),
            "right_eyebrow": get_eyebrow_bounding_box(points["right_eyebrow"], eyebrow_height_scale, eyebrow_width_scale),
            "left_eye": get_eye_bounding_box(points["left_eye"], eye_width_scale, eye_height_scale),
            "right_eye": get_eye_bounding_box(points["right_eye"], eye_width_scale, eye_height_scale),
            "nose": get_nose_bounding_box(points["nose"], nose_width_scale),
            "mouth": get_bounding_box(points["mouth"], mouth_scale)
        }
        for key, bbox in bboxes.items():
            rois[key].append(torch.tensor([i, *bbox]))

    output_dict = {}
    rois_tensors = {}
    for key, roi_list in rois.items():
        rois_tensor = torch.stack(roi_list).to(image_batch.device).float()
        rois_tensors[key] = rois_tensor # 保存RoI张量
        cropped_batch = roi_align(image_batch, rois_tensor, (output_size, output_size), aligned=True)
        output_dict[key] = cropped_batch
        
    return output_dict, rois_tensors