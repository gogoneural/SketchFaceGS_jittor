import cv2
import random
import numpy as np
import argparse

def augment_sketch(img,
                   kernel_sizes=(1, 3, 5),
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
    k = random.choice(kernel_sizes)
    shape = random.choice([
        # cv2.MORPH_RECT,      # 矩形结构元
        # cv2.MORPH_ELLIPSE,   # 椭圆结构元 :contentReference[oaicite:0]{index=0}
        cv2.MORPH_CROSS      # 十字结构元
    ])
    kernel = cv2.getStructuringElement(shape, (k, k))  # 定义结构元 :contentReference[oaicite:1]{index=1}

    # 随机选择迭代次数
    iters = random.choice(iterations)

    # 根据概率决定应用腐蚀或膨胀
    r = random.random()
    if r < p_erosion:
        # 腐蚀：线条变细 :contentReference[oaicite:2]{index=2}
        return cv2.erode(img, kernel, iterations=iters)
    elif r < p_erosion + p_dilation:
        # 膨胀：线条变粗 :contentReference[oaicite:3]{index=3}
        return cv2.dilate(img, kernel, iterations=iters)
    else:
        # 保持原样
        return img

def main():
    parser = argparse.ArgumentParser(description="Sketch Thickness Augmentation")
    parser.add_argument("--input",  "-i", required=True, help="输入线稿图像路径")
    parser.add_argument("--output", "-o", required=True, help="输出增强后图像路径")
    parser.add_argument("--erosion_prob",  type=float, default=1., help="腐蚀概率")
    parser.add_argument("--dilation_prob", type=float, default=0., help="膨胀概率")
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
    kernel_sizes = [3]#list(range(args.min_k, args.max_k + 1))
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