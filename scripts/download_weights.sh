#!/bin/bash
# =============================================================
# SketchFaceGS Weight Downloader
# Usage: bash scripts/download_weights.sh
# =============================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_BASE="https://huggingface.co/Junxiang123/SketchFaceGS_jittor/resolve/main"

download() {
    local url="$1"
    local dst="$2"
    if [ -f "$dst" ]; then
        echo "[skip] $(basename $dst) already exists."
        return
    fi
    mkdir -p "$(dirname "$dst")"
    echo "[download] $(basename $dst) ..."
    wget -q --show-progress -O "$dst" "$url"
}

download "$HF_BASE/model.pkl"            "$REPO_ROOT/checkpoints/model.pkl"
download "$HF_BASE/sketchgen_numpy.pkl"  "$REPO_ROOT/assets/sketchgen_numpy.pkl"
download "$HF_BASE/model_gan_numpy.pkl"  "$REPO_ROOT/assets/model_gan_numpy.pkl"
download "$HF_BASE/FLAME_with_eye_numpy.pkl" "$REPO_ROOT/assets/FLAME_with_eye_numpy.pkl"

echo ""
echo "[done] All weights downloaded."
echo "Run the app with:"
echo "  python app_addsketch_jittor.py --checkpoint checkpoints/model.pkl"
