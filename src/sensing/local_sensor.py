from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalSensingConfig:
    detection_radius_m: float
    max_candidates: int
    position_noise_std_m: float


@dataclass(frozen=True)
class LocalTargetObservation:
    relative_positions: np.ndarray
    mask: np.ndarray


class LocalTargetSensor:
    """Omnidirectional local target sensor with fixed-length candidate output.

    The public observation deliberately contains no target id, role, key flag, or matching
    metadata. Candidates are sorted only by current range and padded with a mask.
    """

    def __init__(self, config: LocalSensingConfig, seed: int | None = None) -> None:
        self.config = config
        if self.config.detection_radius_m < 0.0:
            raise ValueError(f"local_sensing.detection_radius_m must be >= 0, got {self.config.detection_radius_m}.")
        if self.config.max_candidates < 0:
            raise ValueError(f"local_sensing.max_candidates must be >= 0, got {self.config.max_candidates}.")
        if self.config.position_noise_std_m < 0.0:
            raise ValueError(
                "local_sensing.local_position_noise_std_m must be >= 0, "
                f"got {self.config.position_noise_std_m}."
            )

        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    def observe(self, friendly_position: np.ndarray, target_positions: np.ndarray) -> LocalTargetObservation:
        max_candidates = int(self.config.max_candidates)
        relative_positions = np.zeros((max_candidates, 3), dtype=np.float32)
        mask = np.zeros(max_candidates, dtype=np.float32)
        if max_candidates == 0:
            return LocalTargetObservation(relative_positions=relative_positions, mask=mask)

        friendly = np.asarray(friendly_position, dtype=float).reshape(3)
        targets = np.asarray(target_positions, dtype=float)
        if targets.size == 0:
            return LocalTargetObservation(relative_positions=relative_positions, mask=mask)
        if targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError(f"target_positions must have shape (N, 3), got {targets.shape}.")

        true_relative_positions = targets - friendly
        distances = np.linalg.norm(true_relative_positions, axis=1)
        visible_indices = np.flatnonzero(distances <= self.config.detection_radius_m)
        if visible_indices.size == 0:
            return LocalTargetObservation(relative_positions=relative_positions, mask=mask)

        sorted_visible = visible_indices[np.argsort(distances[visible_indices])]
        selected = sorted_visible[:max_candidates]
        observed_relative_positions = true_relative_positions[selected].copy()
        if self.config.position_noise_std_m > 0.0:
            observed_relative_positions = observed_relative_positions + self._rng.normal(
                loc=0.0,
                scale=self.config.position_noise_std_m,
                size=observed_relative_positions.shape,
            )

        count = selected.shape[0]
        relative_positions[:count] = observed_relative_positions.astype(np.float32)
        mask[:count] = 1.0
        return LocalTargetObservation(relative_positions=relative_positions, mask=mask)
