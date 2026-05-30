"""Reproducible seeding across numpy, random, torch, and gymnasium spaces."""
from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Seed the standard library, NumPy, and (if available) PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch not always installed in tests
        pass
