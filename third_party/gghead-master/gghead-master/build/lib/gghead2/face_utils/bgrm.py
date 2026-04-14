# Using MODNet in onnx version, ref: https://github.com/ZHKKKe/MODNet/blob/master/onnx/inference_onnx.py

import onnxruntime
import numpy as np
import cv2
import os

from gghead.env import REPO_ROOT_DIR

session = onnxruntime.InferenceSession(os.path.join(REPO_ROOT_DIR, "assets", "modnet.onnx"), providers=['CUDAExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def get_scale_factor(im_h, im_w, ref_size):

    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32

    x_scale_factor = im_rw / im_w
    y_scale_factor = im_rh / im_h

    return x_scale_factor, y_scale_factor

def modnet(images):
    """
    Run MODNet on a batch of images.
    Returns the alpha matte.
    args:
        images: np.ndarray, shape (B, H, W, 3) in RGB format, values in [0, 255]
    returns:
        np.ndarray, shape (B, H, W) in grayscale, values in [0, 255]
    """
    ref_size = 512
    original_sizes = images.shape[1:3]
    x, y = get_scale_factor(original_sizes[0], original_sizes[1], ref_size)
    images = np.stack([cv2.resize(image, None, fx=x, fy=y, interpolation = cv2.INTER_AREA) for image in images])
    images = images.transpose(0, 3, 1, 2)
    images = (images.astype(np.float32) - 127.5) / 127.5

    outputs = session.run([output_name], {input_name: images})[0]
    outputs = (outputs.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    outputs = np.stack([cv2.resize(output, (original_sizes[1], original_sizes[0]), interpolation = cv2.INTER_AREA) for output in outputs])
    return outputs
    
def mask_image(image, mask):
    """
    Apply a mask to an image.
    args:
        image: np.ndarray, shape (B*, H, W, 3) in RGB format, values in [0, 255]
        mask: np.ndarray, shape (B*, H, W), values in [0, 255]
    """
    mask = mask[..., None] / 255
    return (image * mask + 255 * (1 - mask)).astype(np.uint8)

def remove_background(image):
    """
    Remove the background from an image.
    args:
        image: np.ndarray, shape (H, W, 3) in RGB format, values in [0, 255]
    """
    mask = modnet(image[None])[0]
    return mask_image(image, mask)

def remove_background_batch(images):
    """
    Remove the background from a batch of images.
    args:
        images: np.ndarray, shape (B, H, W, 3) in RGB format, values in [0, 255]
    """
    masks = modnet(images)
    return mask_image(images, masks)