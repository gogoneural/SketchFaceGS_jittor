# sketch_augment_torch.py
import torch
import torch.nn.functional as F
import math
import random

# -------------------- utils --------------------
def rgb_to_gray(x):
    """x: B x 3 x H x W, values in [0,1]
    return: B x 1 x H x W
    """
    r, g, b = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def make_gaussian_kernel(k, sigma, device, dtype):
    """返回 1x1xk xk 的 gaussian kernel tensor"""
    ax = torch.arange(-k // 2 + 1., k // 2 + 1., device=device, dtype=dtype)
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, k, k)

def smooth_field(noise, kernel):
    """
    noise: B x C x h x w
    kernel: 1 x 1 x k x k
    返回: B x C x h x w
    说明：对每个 (B,C) 通道分别做 depthwise conv（kernel 相同）
    """
    B, C, h, w = noise.shape
    ksize = kernel.shape[-1]
    pad = ksize // 2
    kernel = kernel.to(device=noise.device, dtype=noise.dtype)

    # reshape 为 (1, B*C, h, w)，然后做 groups=B*C 的 depthwise conv
    x = noise.view(1, B * C, h, w)
    weight = kernel.expand(B * C, 1, ksize, ksize).contiguous()
    out = F.conv2d(x, weight, bias=None, stride=1, padding=pad, groups=B * C)
    out = out.view(B, C, h, w)
    return out

# -------------------- displacement & warp --------------------
def random_displacement_field(B, H, W, device, dtype, scale=12.0, noise_res=16, sigma=3.0):
    """
    生成平滑位移场（单位：像素），返回 B x 2 x H x W
    - scale: 最大位移幅度（像素）
    - noise_res: 初始噪声低分辨率（越小越平滑）
    - sigma: 高斯平滑尺度
    """
    small_h = max(4, min(noise_res, H))
    small_w = max(4, min(noise_res, W))
    # -1..1 的随机噪声
    noise = (torch.rand(B, 2, small_h, small_w, device=device, dtype=dtype) * 2.0 - 1.0)
    # 上采样到 HxW
    noise_up = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=True)
    # 高斯平滑
    k = int(2 * math.ceil(2.0 * sigma) + 1)
    kernel = make_gaussian_kernel(k, sigma, device=device, dtype=dtype)
    smooth = smooth_field(noise_up, kernel)
    # 缩放为像素位移
    disp = smooth * scale
    return disp

def warp_image_with_field(img, disp):
    """
    img: B x C x H x W
    disp: B x 2 x H x W (dx, dy) in pixels
    returns warped image B x C x H x W
    """
    B, C, H, W = img.shape
    device = img.device
    dtype = img.dtype

    yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                             torch.arange(W, device=device, dtype=dtype),
                             indexing='ij')
    base_x = (xx / (W - 1)) * 2.0 - 1.0  # in [-1,1]
    base_y = (yy / (H - 1)) * 2.0 - 1.0
    base_grid = torch.stack([base_x, base_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B x H x W x 2

    dx = disp[:, 0:1, :, :]  # B x1 H W
    dy = disp[:, 1:2, :, :]
    ndx = dx / ((W - 1) / 2.0)
    ndy = dy / ((H - 1) / 2.0)
    add_grid = torch.cat([ndx, ndy], dim=1).permute(0, 2, 3, 1)  # B x H x W x 2

    sample_grid = base_grid + add_grid
    warped = F.grid_sample(img, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped

# -------------------- morphology --------------------
def dilate_mask(mask, k):
    """mask: Bx1xH xW in {0,1} float. dilation via max_pool2d. k: kernel size (odd)"""
    pad = k // 2
    out = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    out = (out > 0.5).float()
    return out

def erode_mask(mask, k):
    """erosion via neg + max_pool"""
    pad = k // 2
    neg = 1.0 - mask
    neg = F.max_pool2d(neg, kernel_size=k, stride=1, padding=pad)
    out = 1.0 - neg
    out = (out > 0.5).float()
    return out

def random_thickness_torch(mask, min_k=1, max_k=7, prob=1.0):
    """随机膨胀或腐蚀"""
    if random.random() > prob:
        return mask
    op = random.choice(['dilate', 'erode'])
    k = random.randrange(min_k, max_k + 1)
    if k % 2 == 0:
        k += 1
    if op == 'dilate':
        return dilate_mask(mask, k)
    else:
        return erode_mask(mask, k)

# -------------------- dropout / occlusion --------------------
def random_line_dropout_torch(mask, prob=0.2, rect_prob=0.5, max_rect_frac=0.12):
    """
    mask: Bx1xH xW (1=line)
    以 rect 或 圆盘方式删除部分线条
    """
    B, _, H, W = mask.shape
    out = mask.clone()
    for b in range(B):
        if random.random() < prob:
            if random.random() < rect_prob:
                tries = 6
                for _ in range(tries):
                    rw = random.randint(max(1, int(W * 0.03)), max(1, int(W * max_rect_frac)))
                    rh = random.randint(max(1, int(H * 0.03)), max(1, int(H * max_rect_frac)))
                    x = random.randint(0, max(0, W - rw))
                    y = random.randint(0, max(0, H - rh))
                    if out[b, 0, y:y+rh, x:x+rw].sum() > 0:
                        out[b, 0, y:y+rh, x:x+rw] = 0.0
                        break
            else:
                num_pts = random.randint(1, 6)
                ys, xs = (out[b, 0] > 0.5).nonzero(as_tuple=True)
                if ys.numel() == 0:
                    continue
                idxs = torch.randperm(ys.numel())[:num_pts]
                for idx in idxs:
                    yy = int(ys[idx].item())
                    xx = int(xs[idx].item())
                    r = random.randint(1, max(1, min(H, W)//30))
                    y0 = max(0, yy - r)
                    y1 = min(H, yy + r + 1)
                    x0 = max(0, xx - r)
                    x1 = min(W, xx + r + 1)
                    out[b, 0, y0:y1, x0:x1] = 0.0
    return out

# -------------------- pipeline --------------------
def augment_sketch_batch_torch(batch,
                               device=None,
                               p_elastic=0.9, elastic_scale=8.0, elastic_res=16, elastic_sigma=3.0,
                               p_thickness=1.0, min_k=1, max_k=7,
                               p_dropout=0.5, rect_prob=0.5, max_rect_frac=0.12,
                               p_noise=0.6, noise_sigma=0.02):
    """
    batch: B x 3 x H x W, float in [0,1], black lines (dark) on white (bright)
    returns augmented batch same shape device/dtype as input
    """
    assert batch.ndim == 4 and batch.size(1) == 3
    if device is None:
        device = batch.device
    B, C, H, W = batch.shape
    dtype = batch.dtype

    # 0) 灰度化并生成二值线条掩码（1 表示线）
    gray = rgb_to_gray(batch)  # B x1 H W
    masks = torch.zeros_like(gray)
    for b in range(B):
        img = gray[b, 0]
        # 简单阈值：mean * factor（适合黑线白底的线稿）
        t = img.mean().item() * 0.9
        mask = (img < t).float()
        masks[b:b+1] = mask.unsqueeze(0)
    masks = (masks > 0.5).float()

    # 1) 改变粗细
    if random.random() < p_thickness:
        masks = random_thickness_torch(masks, min_k=min_k, max_k=max_k, prob=1.0)

    # 2) 随机删除部分（矩形或点盘）
    if random.random() < p_dropout:
        masks = random_line_dropout_torch(masks, prob=1.0, rect_prob=rect_prob, max_rect_frac=max_rect_frac)

    # 3) 从 mask 重建干净图像（line=0, bg=1）
    clean = (1.0 - masks)  # B x1 H W
    clean3 = clean.repeat(1, 3, 1, 1)  # B x3 H W

    # 4) 弹性位移 warp（对 clean3 做 warp），之后重新二值化以去掉插值灰度
    warped = clean3
    if random.random() < p_elastic:
        disp = random_displacement_field(B, H, W, device=device, dtype=dtype,
                                         scale=elastic_scale, noise_res=elastic_res, sigma=elastic_sigma)
        warped = warp_image_with_field(clean3.to(device=device), disp.to(device=device))
        # rebinarize
        warped_gray = rgb_to_gray(warped)
        for b in range(B):
            t = warped_gray[b].mean().item() * 0.9
            masks_b = (warped_gray[b] < t).float()
            masks[b:b+1] = masks_b
        clean = (1.0 - masks)
        clean3 = clean.repeat(1, 3, 1, 1)
        warped = clean3

    # 5) 小幅噪声（最后一步）
    out = warped
    if random.random() < p_noise:
        noise = torch.randn_like(out) * noise_sigma
        out = out + noise
        out = out.clamp(0.0, 1.0)

    return out

# -------------------- quick test --------------------
if __name__ == "__main__":
    B, C, H, W = 2, 3, 256, 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 构造示例：全白画布 + 几笔黑线
    x = torch.ones(B, C, H, W, device=device, dtype=torch.float32)
    # draw some synthetic strokes (simple rectangles as strokes)
    x[:, :, 60:62, 40:220] = 0.0
    x[:, :, 120:122, 60:220] = 0.0
    x[:, :, 80:200, 100:102] = 0.0
    # run augmentation
    aug = augment_sketch_batch_torch(x, device=device,
                                     p_elastic=0.9, elastic_scale=6.0, elastic_res=12, elastic_sigma=3.0,
                                     p_thickness=1.0, min_k=1, max_k=5,
                                     p_dropout=0.6, rect_prob=0.6, max_rect_frac=0.12,
                                     p_noise=0.6, noise_sigma=0.01)
    print("in:", x.shape, "min/max:", x.min().item(), x.max().item())
    print("out:", aug.shape, "min/max:", aug.min().item(), aug.max().item())
DISPLACEMENT_PRESETS = {
    "low":  dict(p_elastic=0.9, elastic_scale=6.0,  elastic_res=12, elastic_sigma=3.0),
    "med":  dict(p_elastic=0.95, elastic_scale=10.0, elastic_res=24, elastic_sigma=2.5),
    "high": dict(p_elastic=1.0, elastic_scale=18.0, elastic_res=48, elastic_sigma=1.8),
    # 极端 —— 非常大的撕裂式位移（可能产生很强的断裂/错位）
    "extreme": dict(p_elastic=1.0, elastic_scale=30.0, elastic_res=None, elastic_sigma=1.0),
}

def augment_sketch_hard(batch, strength="high", **other_kwargs):
    """
    wrapper: 根据 strength 调整位移参数，产生更大的位移（用于生成错误输入 / 反面数据）
    strength: "low" | "med" | "high" | "extreme"
    other_kwargs: 传入 augment_sketch_batch_torch 的其它参数覆盖默认
    """
    assert strength in DISPLACEMENT_PRESETS
    preset = DISPLACEMENT_PRESETS[strength].copy()

    # 若 elastic_res=None，设为 H（即尽量高频）
    B, C, H, W = batch.shape
    if preset.get("elastic_res", None) is None:
        preset["elastic_res"] = max(8, min(512, max(H, W)))  # cap 防爆内存

    # 合并调用参数（preset 优先，other_kwargs 可覆盖）
    params = {}
    params.update(other_kwargs)
    params.update(preset)

    # 其他保持脚本默认（thickness/dropout/noise 等）
    return augment_sketch_batch_torch(batch, device=batch.device, **params)