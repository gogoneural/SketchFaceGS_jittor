import cv2
import numpy as np
import random
import jittor as jt
def canny_sketch(img):
    edges = cv2.Canny(img, 50, 150)
    return cv2.bitwise_not(edges)  # 颜色反转

def sobel_sketch(img):
    gx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
    absx = cv2.convertScaleAbs(gx)
    absy = cv2.convertScaleAbs(gy)
    merged = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    return cv2.bitwise_not(merged)  # 颜色反转

def laplacian_sketch(img):
    lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    abs_lap = cv2.convertScaleAbs(lap)
    return cv2.bitwise_not(abs_lap)  # 颜色反转

def dog_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    g1 = cv2.GaussianBlur(gray, (0,0), 1.5).astype(np.float32)
    g2 = cv2.GaussianBlur(gray, (0,0), 2.0).astype(np.float32)
    diff = np.abs(g1 - g2)
    diff = cv2.normalize(diff, None, 50, 200, cv2.NORM_MINMAX)
    _, sketch = cv2.threshold(diff.astype(np.uint8), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(sketch)  # 颜色反转


def tensor_to_cv2(tensor):
    """ 将归一化到[-1,1]的B×C×H×W张量转为OpenCV图像
    Args:
        tensor: (B,C,H,W) 范围[-1,1]的float32张量
    Returns:
        cv_img: (H,W,C) BGR格式的uint8 numpy数组
    """
    # 1. 维度转换与反归一化
    if isinstance(tensor, jt.Var):
        img = tensor.permute(1, 2, 0).numpy()
    else:
        img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img  * 255  # [-1,1]→[0,255]

    # 2. 数据类型与通道转换
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB→BGR[3,5](@ref)


def cv2_to_tensor(cv_img):
    """ 将OpenCV图像转为归一化到[-1,1]的张量
    Args:
        cv_img: (H,W,C) BGR格式的uint8 numpy数组
    Returns:
        tensor: (1,C,H,W) 范围[-1,1]的float32张量
    """
    # 1. 通道与数值转换
    # rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # BGR→RGB[3](@ref)
    tensor = jt.array(cv_img.astype(np.float32) / 255.)

    # 2. 维度扩展与重排
    return tensor[None,...] # HWC→BCHW[7,8](@ref)


def random_sketch_extractor(image,method=None):
    img_ = []
    for i in range(image.shape[0]):
        img = tensor_to_cv2(image[i])
        # assert img is not None, f"无法读取图像：{path}"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (5,5), 0)
        methods = {
            'canny': canny_sketch,
            'sobel': sobel_sketch,
            'laplacian': laplacian_sketch,
            'dog': dog_sketch,
        }
        if method == None:
            choice = random.choice(list(methods.keys()))
        else:
            choice = method
        # print(f"使用方法: {choice}")
        img =  methods[choice](gray)
        img_.append(cv2_to_tensor(img))
    image = jt.stack(img_,0).repeat(1,3,1,1)
    return image
if __name__ == "__main__":
    sketch = random_sketch_extractor("f_image.png")
    cv2.imwrite("sketch_output.png", sketch)
    print("白底黑线素描已保存")