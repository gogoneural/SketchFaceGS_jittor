from eg3d.metrics import frechet_inception_distance
from eg3d.metrics.metric_main import register_metric
from gghead_jittor.dataset.image_folder_dataset import (
    GGHeadMaskImageFolderDataset,
    GGHeadImageFolderDatasetConfig,
)

import os
# import dnnlib


# def _patch_dnnlib_open_url_for_inception():
#     """如果本地存在 inception-2015-12-05.pkl，则把 dnnlib.util.open_url 打补丁，返回本地文件句柄。

#     检查的候选路径（按顺序）：
#     - 环境变量 `INCEPTION_PKL_PATH` 指定的路径
#     - 当前工作目录下的 `inception-2015-12-05.pkl`
#     - 用户缓存目录 `~/.cache/inception-2015-12-05.pkl`
#     - 用户 dnnlib 缓存 `~/.cache/dnnlib/inception-2015-12-05.pkl`
#     - 仓库相对的 `src/gghead/weights/inception-2015-12-05.pkl`

#     一旦找到第一个存在的文件，会替换 `dnnlib.util.open_url`，仅在请求以该文件名结尾时返回本地文件句柄，其他 URL 使用原实现。
#     """

#     candidates = []
#     env_path = os.environ.get("INCEPTION_PKL_PATH")
#     if env_path:
#         candidates.append(env_path)
#     candidates.extend(
#         [
#             os.path.join(os.getcwd(), "inception-2015-12-05.pkl"),
#             os.path.expanduser("~/.cache/inception-2015-12-05.pkl"),
#             os.path.expanduser("~/.cache/dnnlib/inception-2015-12-05.pkl"),
#             os.path.join(
#                 os.path.dirname(__file__), "..", "weights", "inception-2015-12-05.pkl"
#             ),
#         ]
#     )

#     for p in candidates:
#         try:
#             if p and os.path.exists(p):
#                 orig_open_url = dnnlib.util.open_url

#                 def _open_url_wrapper(url, *args, _local_path=p, **kwargs):
#                     if isinstance(url, str) and url.endswith(
#                         "inception-2015-12-05.pkl"
#                     ):
#                         return open(_local_path, "rb")
#                     return orig_open_url(url, *args, **kwargs)

#                 dnnlib.util.open_url = _open_url_wrapper
#                 print(f"[gghead] using local inception pkl: {p}")
#                 return
#         except Exception:
#             # 保守处理：如果检查路径时出错，继续尝试下一个候选项
#             continue


# 在模块导入时尝试打补丁，使后续 compute_fid 调用（在离线环境中）能使用本地 pkl
# _patch_dnnlib_open_url_for_inception()


@register_metric
def fid100(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.dataset = GGHeadMaskImageFolderDataset(
        GGHeadImageFolderDatasetConfig(**opts.dataset_kwargs)
    )
    fid = frechet_inception_distance.compute_fid(opts, max_real=100, num_gen=100)
    return dict(fid100=fid)


@register_metric
def fid1k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.dataset = GGHeadMaskImageFolderDataset(
        GGHeadImageFolderDatasetConfig(**opts.dataset_kwargs)
    )
    fid = frechet_inception_distance.compute_fid(opts, max_real=1000, num_gen=1000)
    return dict(fid1k=fid)


# @register_metric
# def fid1k_broken(opts):
#     opts.dataset_kwargs.update(max_size=None, xflip=False)
#     opts.dataset = GGHMaskImageFolderDataset(GGHImageFolderDatasetConfig(**opts.dataset_kwargs))
#     fid = frechet_inception_distance.compute_fid(opts, max_real=250, num_gen=1000)
#     return dict(fid1k=fid)


@register_metric
def fid5k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.dataset = GGHeadMaskImageFolderDataset(
        GGHeadImageFolderDatasetConfig(**opts.dataset_kwargs)
    )
    fid = frechet_inception_distance.compute_fid(opts, max_real=5000, num_gen=5000)
    return dict(fid5k=fid)


@register_metric
def fid10k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.dataset = GGHeadMaskImageFolderDataset(
        GGHeadImageFolderDatasetConfig(**opts.dataset_kwargs)
    )
    fid = frechet_inception_distance.compute_fid(opts, max_real=10000, num_gen=10000)
    return dict(fid10k=fid)


@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.dataset = GGHeadMaskImageFolderDataset(
        GGHeadImageFolderDatasetConfig(**opts.dataset_kwargs)
    )
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)
