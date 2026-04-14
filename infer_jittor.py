#!/usr/bin/env python3
"""
Jittor inference for SketchFaceGS (non-fusion / pure generation).

Usage:
  python infer_jittor.py \
      --checkpoint checkpoints/model.pkl \
      --config configs/train.yaml \
      --f_image examples/color_1.png --sketch examples/sketch_1.png \
      --out_dir outputs/infer_jittor
"""
import os
import sys
import argparse
import pickle
import numpy as np
from PIL import Image

import jittor as jt
from jittor import transform

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_base_dir = os.path.dirname(os.path.abspath(__file__))
_third_party_path = os.path.join(_base_dir, "third_party")
_gghead_src = os.path.join(_third_party_path, "gghead-master", "gghead-master", "src")
_lhm_path = os.path.join(_third_party_path, "LHM")
_jg_path = os.path.join(_third_party_path, "JGaussian-main")
for _p in [_third_party_path, _lhm_path, _gghead_src, _jg_path]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.models.model_jittor import SketchFaceGS
from src.utils.utils import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_image(path, size=512):
    img = Image.open(path).convert("RGB")
    tfm = transform.Compose([
        transform.Resize((size, size)),
        transform.ToTensor(),
    ])
    t = tfm(img)
    if not isinstance(t, jt.Var):
        t = jt.array(t)
    return t.unsqueeze(0)  # [1, 3, H, W] in [0, 1]


def save_tensor(tensor, filename):
    """Save a [B, C, H, W] tensor as an image (first sample)."""
    tensor = jt.clamp(tensor, 0.0, 1.0)
    img_np = tensor[0].numpy().transpose(1, 2, 0)
    img_np = (img_np * 255.0).astype(np.uint8)
    if img_np.shape[2] == 1:
        img_np = img_np.squeeze(-1)
    Image.fromarray(img_np).save(filename)


def remap_ckpt_key(k: str) -> str:
    """Map Torch checkpoint keys to Jittor module names when APIs differ."""
    return (
        k.replace(".attn.norm_added_q.", ".attn.norm_q.")
         .replace(".attn.norm_added_k.", ".attn.norm_k.")
         .replace(".ff.net.0.proj.", ".ff.net.0.0.")
         .replace(".ff_context.net.0.proj.", ".ff_context.net.0.0.")
    )


def resolve_repo_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(_base_dir, path)


def resolve_checkpoint_path(path):
    resolved = resolve_repo_path(path)
    default_checkpoint = os.path.normpath("checkpoints/model.pkl")
    legacy_checkpoints = [
        os.path.join(_base_dir, "output", "model.pkl"),
        os.path.join(_base_dir, "output", "new_jittor_remap_v2.pkl"),
        os.path.join(_base_dir, "output", "new_jittor_remap.pkl"),
    ]
    if os.path.normpath(path) == default_checkpoint and not os.path.exists(resolved):
        for legacy_checkpoint in legacy_checkpoints:
            if os.path.exists(legacy_checkpoint):
                return legacy_checkpoint
    return resolved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SketchFaceGS Jittor Inference")
    parser.add_argument("--checkpoint", default="checkpoints/model.pkl", type=str,
                        help="Path to numpy pkl checkpoint (default: checkpoints/model.pkl)")
    parser.add_argument("--config", default="configs/train.yaml", type=str,
                        help="Training config yaml (default: configs/train.yaml)")
    parser.add_argument("--f_image", type=str, default="examples/color_1.png",
                        help="Face image for color reference")
    parser.add_argument("--sketch", type=str, default="examples/sketch_1.png",
                        help="Sketch image for shape guidance")
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs/infer_jittor")
    parser.add_argument("--precision", choices=["fp32", "amp"], default="fp32")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for generation")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    args.checkpoint = resolve_checkpoint_path(args.checkpoint)
    args.config = resolve_repo_path(args.config)
    args.f_image = resolve_repo_path(args.f_image)
    args.sketch = resolve_repo_path(args.sketch)
    args.mask = resolve_repo_path(args.mask)
    args.out_dir = resolve_repo_path(args.out_dir)

    for _label, _path in [
        ("checkpoint", args.checkpoint),
        ("config", args.config),
        ("f_image", args.f_image),
        ("sketch", args.sketch),
    ]:
        if _path is not None and not os.path.exists(_path):
            raise FileNotFoundError(f"Missing {_label}: {_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Device ----
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    if args.precision == "amp":
        jt.flags.use_amp = 1

    # ---- Config ----
    cfg = load_config(args.config)

    # ---- Build model ----
    print("Building model...")
    model = SketchFaceGS(model_cfg=cfg.MODEL)

    # ---- Load checkpoint (numpy pkl, no torch needed) ----
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        state_np = pickle.load(f)

    # Custom loader: Jittor's load_parameters chokes on None values.
    # We also need to handle buffers that aren't nn.Parameters.
    model_sd = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in state_np.items():
        if v is None:
            skipped += 1
            continue
        target_k = k if k in model_sd else remap_ckpt_key(k)
        if target_k in model_sd:
            try:
                param = model_sd[target_k]
                arr = jt.array(v) if not isinstance(v, jt.Var) else v
                if param.shape == arr.shape:
                    param.update(arr)
                    loaded += 1
                else:
                    print(f"  [SKIP shape] {k} -> {target_k}: model {param.shape} vs ckpt {arr.shape}")
                    skipped += 1
            except Exception as e:
                print(f"  [SKIP error] {k} -> {target_k}: {e}")
                skipped += 1
        else:
            # Try setattr for buffers not in state_dict
            parts = k.split(".")
            obj = model
            try:
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                setattr(obj, parts[-1], jt.array(v) if not isinstance(v, jt.Var) else v)
                loaded += 1
            except (AttributeError, TypeError):
                skipped += 1

    model.eval()
    print(f"  Loaded {loaded} params, skipped {skipped}")

    # ---- Inputs ----
    f_img_t = None
    if args.f_image is not None:
        f_img_t = load_image(args.f_image, size=args.size)
        print(f"  f_image: {args.f_image}")

    sketch_t = None
    if args.sketch is not None:
        sketch_t = load_image(args.sketch, size=args.size)
        print(f"  sketch:  {args.sketch}")

    mask_t = None
    if args.mask is not None:
        mask_t = load_image(args.mask, size=args.size)[:, [0]]

    # ---- Inference (non-fusion) ----
    print("Running inference (non-fusion)...")
    from src.utils.camera_utils import rand_c2w
    cs_main = rand_c2w(model.cam_pivot, 1, alpha_deg=0.0, beta_deg=0.0)
    with jt.no_grad():
        results = model(
            batch_size=1,
            idx=args.seed,
            sketch_img=sketch_t,
            f_image=f_img_t,
            mask=mask_t,
            fusion=False,
            cs_in=cs_main,
        )

    # ---- Save outputs ----
    for name in ["gen_image", "feedforward_image", "adain_image"]:
        if results.get(name) is not None:
            out_path = os.path.join(args.out_dir, f"{name}.png")
            save_tensor(results[name], out_path)
            print(f"  Saved {name} -> {out_path}")

    if results.get("f_image") is not None:
        save_tensor(results["f_image"], os.path.join(args.out_dir, "f_image.png"))
    if results.get("t_image") is not None:
        save_tensor(results["t_image"], os.path.join(args.out_dir, "t_image.png"))

    # ---- Multi-view rendering ----
    gs_model = results["conditions"]["gs_model"]
    if gs_model is not None:
        import math

        n_views = 8
        # Generate evenly-spaced horizontal angles: -60° to +60° with slight vertical variation
        cam_pivot = jt.array([0, 0.05, 0.2]).float32()
        cs_list = []
        for i in range(n_views):
            alpha = -60 + 120 * i / (n_views - 1)  # horizontal: -60° to +60°
            beta = 0.0  # eye-level
            c_i = rand_c2w(cam_pivot, 1, alpha_deg=alpha, beta_deg=beta)
            cs_list.append(c_i)
        cs_selected = jt.concat(cs_list, dim=0)
        print(f"  Rendering {n_views} multi-view images (horizontal -60° to +60°)...")
        with jt.no_grad():
            mv_images = model.gs_gen(gs_model=gs_model, c=cs_selected)
        for vi in range(mv_images.shape[0]):
            out_path = os.path.join(args.out_dir, f"view_{vi:02d}.png")
            save_tensor(mv_images[vi:vi+1], out_path)
        print(f"  Saved {n_views} views to {args.out_dir}/view_*.png")

    print(f"Done! All outputs in {args.out_dir}")


if __name__ == "__main__":
    main()