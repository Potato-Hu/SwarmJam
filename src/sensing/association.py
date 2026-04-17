from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AssociationOutput:
    """Placeholder output for a future target association module."""

    target_est_relative_position: np.ndarray
    confidence: float
    valid: bool


def empty_association_output() -> AssociationOutput:
    return AssociationOutput(
        target_est_relative_position=np.zeros(3, dtype=np.float32),
        confidence=0.0,
        valid=False,
    )
