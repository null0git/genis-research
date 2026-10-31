"""Reproducible random-seed management.

Every experiment script calls set_all_seeds() once, at the start of each
run/fold/seed iteration, so that numpy, Python's random module, and torch
(CPU and, if available, CUDA) all share the same fixed seed. This does not
guarantee bit-exact reproduction across different hardware or library
versions (see ENVIRONMENT.md) but it removes run-to-run randomness on the
same machine.
"""

from __future__ import annotations

import random
import os

import numpy as np


def set_all_seeds(seed: int) -> None:
    """Set the random seed for python, numpy, and torch (if installed).

    Args:
        seed: Integer seed to apply everywhere.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms where possible; some ops fall back with a
        # warning rather than an error if a deterministic kernel does not
        # exist, which is acceptable for this project's purposes.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        # torch is only required for the MLP/1D-CNN/GRU models; tree-based
        # experiments can run without it.
        pass
