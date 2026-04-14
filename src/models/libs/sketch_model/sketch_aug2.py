import cv2
import random
import numpy as np
import argparse
import jittor as jt
def augment_sketch(img,
                   dilation_kernel_sizes,
                   erosion_kernel_sizes,
                   p_erosion=0.3,
                   p_dilation=0.3,
                   iterations=(1, 2)):
    """
    对单通道二值线稿图像进行随机膨胀或腐蚀增强。

    :param img: np.ndarray, H×W 单通道（0 或 255）
    :param kernel_sizes: 结构元尺寸列表，单位：像素
    :param p_erosion: 执行腐蚀的概率
    :param p_dilation: 执行膨胀的概率
    :param iterations: 腐蚀/膨胀的迭代次数范围
    :return: 增强后的图像
    """
    # 随机选择结构元大小与形状
    # k = random.choice(kernel_sizes)
    shape = random.choice([
        cv2.MORPH_RECT,      # 矩形结构元
        cv2.MORPH_ELLIPSE,   # 椭圆结构元 :contentReference[oaicite:0]{index=0}
        cv2.MORPH_CROSS      # 十字结构元
    ])
    # kernel = cv2.getStructuringElement(shape, (k, k))  # 定义结构元 :contentReference[oaicite:1]{index=1}

    # 随机选择迭代次数
    iters = random.choice(iterations)

    # 根据概率决定应用腐蚀或膨胀
    r = random.random()
    if r < p_erosion:
        k = random.choice(erosion_kernel_sizes)
        kernel = cv2.getStructuringElement(shape, (k, k))
        # 腐蚀：线条变细 :contentReference[oaicite:2]{index=2}
        return cv2.erode(img, kernel, iterations=iters)
    elif r < p_erosion + p_dilation:
        # 膨胀：线条变粗 :contentReference[oaicite:3]{index=3}
        k = random.choice(dilation_kernel_sizes)
        kernel = cv2.getStructuringElement(shape, (k, k))
        return cv2.dilate(img, kernel, iterations=iters)
    else:
        # 保持原样
        return img
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
    img = (img * 0.5 + 0.5) * 255  # [-1,1]→[0,255]

    # 2. 数据类型与通道转换
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # RGB→BGR[3,5](@ref)


def cv2_to_tensor(cv_img):
    """ 将OpenCV图像转为归一化到[-1,1]的张量
    Args:
        cv_img: (H,W,C) BGR格式的uint8 numpy数组
    Returns:
        tensor: (1,C,H,W) 范围[-1,1]的float32张量
    """
    # 1. 通道与数值转换
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # BGR→RGB[3](@ref)
    tensor = jt.array(rgb.astype(np.float32) / 127.5 - 1.0)

    # 2. 维度扩展与重排
    return tensor.permute(2,0,1)  # HWC→BCHW[7,8](@ref)

def sketch_aug2(image):
    img_ = []
    for i in range(image.shape[0]):
        img = tensor_to_cv2(image[i])
        dilation_kernel_sizes = [1, 2, 3, 4] 
        erosion_kernel_sizes = [1, 2, ] # list(range(args.min_k, args.max_k + 1))
        img = augment_sketch(
            img,
            dilation_kernel_sizes=erosion_kernel_sizes,
            erosion_kernel_sizes=dilation_kernel_sizes,#反的
            p_erosion=0.7,
            p_dilation=0.3,
            iterations=(1,2)
        )
        img_.append(cv2_to_tensor(img))
    image = jt.stack(img_,0)
    return image





def main():
    parser = argparse.ArgumentParser(description="Sketch Thickness Augmentation")
    parser.add_argument("--input",  "-i", required=True, help="输入线稿图像路径")
    parser.add_argument("--output", "-o", required=True, help="输出增强后图像路径")
    parser.add_argument("--erosion_prob",  type=float, default=0.3, help="腐蚀概率")
    parser.add_argument("--dilation_prob", type=float, default=0.3, help="膨胀概率")
    parser.add_argument("--min_k", type=int, default=1, help="结构元最小尺寸")
    parser.add_argument("--max_k", type=int, default=5, help="结构元最大尺寸")
    parser.add_argument("--iters", nargs=2, type=int, default=(1,2),
                        help="迭代次数范围，例如：1 2")
    args = parser.parse_args()

    # 读取灰度图并二值化
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"错误：无法读取图像 {args.input}")
        return
    # _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)  # 二值化

    # 调用增强函数
    kernel_sizes = [1,2,3]#list(range(args.min_k, args.max_k + 1))
    out = augment_sketch(
        img,
        kernel_sizes=kernel_sizes,
        p_erosion=args.erosion_prob,
        p_dilation=args.dilation_prob,
        iterations=tuple(args.iters)
    )

    # 保存结果
    cv2.imwrite(args.output, out)
    print(f"增强后的图像已保存至 {args.output}")

if __name__ == "__main__":
    main()