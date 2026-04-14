#!/bin/bash
# ============================================================================
# SketchFaceGS Jittor Inference Script
# ============================================================================
# Usage:
#   bash run_inference_jittor.sh [optional infer_jittor.py args]
#
# Example:
#   bash run_inference_jittor.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Jittor environment hints
export JT_USE_CUDA=1

python infer_jittor.py "$@"
