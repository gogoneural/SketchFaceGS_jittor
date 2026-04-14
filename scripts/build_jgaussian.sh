#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JGAUSSIAN_OPS_DIR="$REPO_ROOT/third_party/JGaussian-main/ops"

if [ ! -d "$JGAUSSIAN_OPS_DIR" ]; then
    echo "[error] JGaussian ops directory not found: $JGAUSSIAN_OPS_DIR"
    exit 1
fi

build_dir() {
    local dir="$1"
    local required="${2:-1}"
    echo "[build] $dir"
    if [ -f "$dir/setup.py" ]; then
        if (cd "$dir" && python setup.py install); then
            return
        fi
        if [ "$required" = "1" ]; then
            echo "[error] failed to build required op: $dir"
            return 1
        fi
        echo "[warn] failed to build optional op: $dir"
        return 0
    fi
    if [ -f "$dir/Makefile" ]; then
        if make -C "$dir"; then
            return
        fi
        if [ "$required" = "1" ]; then
            echo "[error] failed to build required op: $dir"
            return 1
        fi
        echo "[warn] failed to build optional op: $dir"
        return 0
    fi
    echo "[skip] no setup.py or Makefile in $dir"
}

build_dir "$JGAUSSIAN_OPS_DIR/diff_gaussian_rasterization" 1
build_dir "$JGAUSSIAN_OPS_DIR/diff_surfel_rasterization" 1
build_dir "$JGAUSSIAN_OPS_DIR/mip_diff_gaussian_rasterizater" 1
build_dir "$JGAUSSIAN_OPS_DIR/simple_knn" 1

if [ -d "$JGAUSSIAN_OPS_DIR/ACAP" ]; then
    build_dir "$JGAUSSIAN_OPS_DIR/ACAP" 0
fi

echo "[done] JGaussian ops built successfully."
