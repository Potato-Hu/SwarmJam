from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set the shared random seed for Python and NumPy."""

    random.seed(seed)
    np.random.seed(seed)


def make_rng(
    seed: int | None = None,
    *,
    offset: int = 0,
    fallback: int = 0,
) -> np.random.Generator:
    """Create a reproducible NumPy random generator."""

    base_seed = fallback if seed is None else seed
    return np.random.default_rng(int(base_seed) + int(offset))
