from __future__ import annotations

import torch


class ValueNormalizer:
    """Running mean/std normalizer for MAPPO value targets.

    The critic predicts normalized values. During rollout/bootstrap we denormalize the
    critic output back to the environment reward scale before computing GAE. During
    critic training we normalize the return targets again, so the value loss stays on a
    stable scale even if raw returns drift after reward/observation changes.
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.epsilon = float(epsilon)
        self.count = float(epsilon)
        self.mean = 0.0
        self.var = 1.0

    @property
    def std(self) -> float:
        return float(max(self.var, 1e-12) ** 0.5)

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        detached = values.detach().float().reshape(-1)
        batch_mean = float(detached.mean().item())
        batch_var = float(detached.var(unbiased=False).item())
        batch_count = float(detached.numel())
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean

    def state_dict(self) -> dict[str, float]:
        return {
            "count": float(self.count),
            "mean": float(self.mean),
            "var": float(self.var),
            "epsilon": float(self.epsilon),
        }

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.count = float(state.get("count", self.epsilon))
        self.mean = float(state.get("mean", 0.0))
        self.var = float(state.get("var", 1.0))
        self.epsilon = float(state.get("epsilon", self.epsilon))

    def _update_from_moments(self, batch_mean: float, batch_var: float, batch_count: float) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = delta * delta * self.count * batch_count / total_count
        new_var = (m_a + m_b + correction) / total_count

        self.mean = float(new_mean)
        self.var = float(max(new_var, 1e-12))
        self.count = float(total_count)
