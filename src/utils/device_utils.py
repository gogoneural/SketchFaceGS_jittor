"""Device selection helpers for single / multi-GPU execution (Jittor version).

This module provides a small utility to select the appropriate CUDA device
for Jittor-based execution.
"""

from __future__ import annotations

import os
import jittor as jt


def get_best_cuda_device() -> int:
    """Return a CUDA device index chosen for multi-GPU or single-GPU runs.

    Selection priority:
    - If CUDA unavailable -> -1 (CPU)
    - If `LOCAL_RANK` env var present -> use it
    - Else if `RANK` present -> use `RANK % num_gpus`
    - Else default to device 0

    Also enables CUDA in Jittor flags.
    """
    if not jt.has_cuda:
        jt.flags.use_cuda = 0
        return -1

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        rank = os.environ.get("RANK")
        try:
            if rank is not None:
                idx = int(rank) % jt.get_device_count()
            else:
                idx = 0
        except Exception:
            idx = 0
    else:
        try:
            idx = int(local_rank)
        except Exception:
            idx = 0

    ngpu = jt.get_device_count()
    if ngpu == 0:
        jt.flags.use_cuda = 0
        return -1
    idx = idx % ngpu
    jt.flags.use_cuda = 1
    return idx
