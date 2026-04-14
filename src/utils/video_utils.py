import numpy as np
import jittor as jt
import math
from src.utils.camera_utils import LookAtPoseSampler
from typing import Optional
def get_cs_list(c, device=None):
    num_frames = 96
    # 动画阶段划分
    trans_frames = 18
    rot_frames = 60
    
    # 控制"圆"的大小
    max_circle_radius = 0.5  # 离开圆心的距离，越大晃动感越强
    cam_pivot = jt.array([0, 0.05, 0.2]).float32()  # 头部中心

    # 1. 解析初始相机矩阵
    # c: [pose(16) | intrinsics(9)]
    initial_pose = c[0, :16].reshape(4, 4)
    initial_pos = initial_pose[:3, 3]
    
    # 提取当前相机的局部坐标系 (用于构建圆周运动所在的平面)
    cam_right = initial_pose[:3, 0]
    cam_up = initial_pose[:3, 1]
    
    cs_list = []

    for i in range(num_frames):
        # 计算当前半径和角度
        if i < trans_frames:
            # 阶段1: 从中心向外扩散
            t = i / trans_frames
            t = (1 - math.cos(t * math.pi)) / 2  # 平滑
            curr_radius = t * max_circle_radius
            angle = t * (2 * math.pi * 0.2)  # 转一点点
        elif i < trans_frames + rot_frames:
            # 阶段2: 保持半径进行 360 度旋转
            t = (i - trans_frames) / rot_frames
            curr_radius = max_circle_radius
            angle = (2 * math.pi * 0.2) + (t * 2 * math.pi)
        else:
            # 阶段3: 收回到中心
            t = (i - trans_frames - rot_frames) / trans_frames
            t = (1 - math.cos(t * math.pi)) / 2
            curr_radius = (1 - t) * max_circle_radius
            angle = (2 * math.pi * 0.2) + (2 * math.pi) + t * (2 * math.pi * 0.2)

        # 2. 计算在当前视角平面上的偏移
        offset = (math.cos(angle) * cam_right + math.sin(angle) * cam_up) * curr_radius
        new_pos = initial_pos + offset

        # 3. 使用 LookAtPoseSampler 重新生成姿态
        relative_pos = new_pos - cam_pivot
        r = jt.norm(relative_pos)
        
        # 计算水平角 (yaw) 和 垂直角 (pitch)
        h = float(jt.arctan2(relative_pos[0], relative_pos[2]).item())
        v = float(jt.asin(relative_pos[1] / r).item())

        pose = LookAtPoseSampler.sample(
            horizontal_mean=h,
            vertical_mean=v,
            lookat_position=cam_pivot,
            radius=float(r.item()),
            horizontal_stddev=0,
            vertical_stddev=0,
            batch_size=1,
        ).reshape(-1, 16)

        # 4. 保持原始内参
        intrinsics = c[:, 16:] 
        c_frame = jt.concat([pose, intrinsics], dim=1)
        cs_list.append(c_frame)

    return jt.concat(cs_list, dim=0)


# Legacy main() removed — was torch-only code not used by Jittor inference pipeline