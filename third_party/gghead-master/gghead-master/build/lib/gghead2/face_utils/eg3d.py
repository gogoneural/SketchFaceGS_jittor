# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch
import cv2
from PIL import Image
from gghead.face_utils.preprocess import align_img
from gghead.face_utils.load_mats import load_lm3d
from gghead.face_utils.deca_onnx import CropFace
from gghead.env import REPO_ROOT_DIR

lm3d_std = load_lm3d(os.path.join(REPO_ROOT_DIR, "assets", "BFM"))
crop_face = CropFace()

def crop_image(image):
    im = Image.fromarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    lm = crop_face.predict_landmarks68(image)
    _, H = im.size
    lm = lm.reshape((-1, 2))
    lm[:, -1] = H - 1 - lm[:, -1]

    target_size = 1024.
    rescale_factor = 300
    center_crop_size = 700
    output_size = 512

    _, im_high, _, _, = align_img(im, lm, lm3d_std, target_size=target_size, rescale_factor=rescale_factor)

    left = int(im_high.size[0]/2 - center_crop_size/2)
    upper = int(im_high.size[1]/2 - center_crop_size/2)
    right = left + center_crop_size
    lower = upper + center_crop_size
    im_cropped = im_high.crop((left, upper, right,lower))
    im_cropped = im_cropped.resize((output_size, output_size), resample=Image.LANCZOS)
    image_cropped = np.array(im_cropped)
    return image_cropped