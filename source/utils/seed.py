"""
Seed fixing utility for reproducible experiments.

Call seed_everything(seed) before EACH training run to ensure:
  - Same weight initialization
  - Same dropout masks
  - Same batch ordering
  - Same mixup augmentation

Usage in __main__.py:
    from source.utils.seed import seed_everything

    for run_idx in range(run_count):
        seed = base_seed + run_idx   # different but FIXED seed per run
        seed_everything(seed)
        training = model_training(cfg)
        ...
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True):
    """Fix ALL random seeds for reproducibility.

    Args:
        seed: random seed
        deterministic: if True, use deterministic CUDA algorithms
            (slightly slower but fully reproducible)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Must set BEFORE calling use_deterministic_algorithms
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Some ops don't have deterministic impl — fall back
                torch.use_deterministic_algorithms(False)
    else:
        torch.backends.cudnn.benchmark = True