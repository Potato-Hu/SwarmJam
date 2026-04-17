from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GlobalKeyTargetSensingConfig:
    radar_delay_seconds: float
    position_noise_std_m: float


class DelayedNoisyKeyTargetSensor:
    """External global sensor for key-target positions.

    It observes delayed key-target truth and adds small position noise. When the requested
    delay reaches before the available history, the earliest available truth is used.
    """

    def __init__(self, config: GlobalKeyTargetSensingConfig, seed: int | None = None) -> None:
        self.config = config
        if self.config.radar_delay_seconds < 0.0:
            raise ValueError(
                "global_sensing.radar_delay_seconds must be >= 0, "
                f"got {self.config.radar_delay_seconds}."
            )
        if self.config.position_noise_std_m < 0.0:
            raise ValueError(
                "global_sensing.key_enemy_position_noise_std_m must be >= 0, "
                f"got {self.config.position_noise_std_m}."
            )

        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._truth_history: list[np.ndarray] = []
        self._cache: np.ndarray | None = None
        self._cache_step: int | None = None

    def reset(self, initial_key_positions: np.ndarray, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._truth_history = [np.asarray(initial_key_positions, dtype=float).copy()]
        self._cache = None
        self._cache_step = None

    def record_truth(self, key_positions: np.ndarray) -> None:
        self._truth_history.append(np.asarray(key_positions, dtype=float).copy())
        self._cache = None
        self._cache_step = None

    def observe(self, *, time_step: int, dt: float, fallback_key_positions: np.ndarray) -> np.ndarray:
        if self._cache_step == int(time_step) and self._cache is not None:
            return self._cache.copy()

        if not self._truth_history:
            self._truth_history = [np.asarray(fallback_key_positions, dtype=float).copy()]

        delay_steps = int(round(self.config.radar_delay_seconds / float(dt))) if dt > 0.0 else 0
        history_idx = max(0, len(self._truth_history) - 1 - delay_steps)
        observed_positions = self._truth_history[history_idx].copy()
        if observed_positions.size and self.config.position_noise_std_m > 0.0:
            noise = self._rng.normal(
                loc=0.0,
                scale=self.config.position_noise_std_m,
                size=observed_positions.shape,
            )
            observed_positions = observed_positions + noise

        self._cache = observed_positions.astype(float, copy=True)
        self._cache_step = int(time_step)
        return self._cache.copy()
