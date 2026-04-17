"""Space builders for the swarm RL project.

This file defines action, observation, and global-state spaces used by the env skeleton.
Current observation design keeps a compact geometry layout over the policy-visible enemy
view. Observation mode exposes key targets only; groundtruth mode keeps the full enemy list.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    from gymnasium import spaces as gym_spaces
except ModuleNotFoundError:
    @dataclass(frozen=True)
    class _FallbackDiscrete:
        n: int

    @dataclass(frozen=True)
    class _FallbackBox:
        low: np.ndarray
        high: np.ndarray
        shape: tuple[int, ...]
        dtype: Any

    gym_spaces = SimpleNamespace(Discrete=_FallbackDiscrete, Box=_FallbackBox)

spaces = gym_spaces


@dataclass(frozen=True)
class SwarmSpaceSpec:
    """Compact summary of the current multi-agent space layout."""

    num_friendlies: int
    num_enemies: int
    per_agent_action_dim: int
    per_agent_obs_dim: int
    global_state_dim: int


def build_agent_ids(num_friendlies: int) -> list[str]:
    return [f"agent_{idx}" for idx in range(num_friendlies)]


def build_action_spaces(num_friendlies: int, action_dim: int) -> dict[str, Any]:
    return {agent_id: spaces.Discrete(action_dim) for agent_id in build_agent_ids(num_friendlies)}


def build_observation_spaces(num_friendlies: int, num_enemies: int, num_key_enemies: int, bounds: np.ndarray) -> dict[str, Any]:
    assigned_target_dim = 4 + num_key_enemies if num_key_enemies > 0 else 0
    per_agent_obs_dim = 3 + 4 * num_enemies + num_key_enemies + assigned_target_dim + 3 * num_friendlies
    high = np.ones(per_agent_obs_dim, dtype=np.float32)
    low = -np.ones(per_agent_obs_dim, dtype=np.float32)
    return {
        agent_id: spaces.Box(low=low, high=high, shape=(per_agent_obs_dim,), dtype=np.float32)
        for agent_id in build_agent_ids(num_friendlies)
    }


def build_global_state_space(num_friendlies: int, num_enemies: int, num_key_enemies: int, bounds: np.ndarray) -> Any:
    global_state_dim = 3 * num_enemies + 3 * num_friendlies + num_key_enemies + num_friendlies
    position_high = np.full(3 * num_enemies + 3 * num_friendlies, np.max(bounds), dtype=np.float32)
    power_high = np.full(num_key_enemies + num_friendlies, np.finfo(np.float32).max, dtype=np.float32)
    high = np.concatenate([position_high, power_high], dtype=np.float32)
    low = np.concatenate(
        [
            np.zeros(3 * num_enemies + 3 * num_friendlies, dtype=np.float32),
            np.full(num_key_enemies + num_friendlies, -np.finfo(np.float32).max, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    return spaces.Box(low=low, high=high, shape=(global_state_dim,), dtype=np.float32)


def build_space_spec(num_friendlies: int, num_enemies: int, num_key_enemies: int, action_dim: int) -> SwarmSpaceSpec:
    assigned_target_dim = 4 + num_key_enemies if num_key_enemies > 0 else 0
    per_agent_obs_dim = 3 + 4 * num_enemies + num_key_enemies + assigned_target_dim + 3 * num_friendlies
    global_state_dim = 3 * num_enemies + 3 * num_friendlies + num_key_enemies + num_friendlies
    return SwarmSpaceSpec(
        num_friendlies=num_friendlies,
        num_enemies=num_enemies,
        per_agent_action_dim=action_dim,
        per_agent_obs_dim=per_agent_obs_dim,
        global_state_dim=global_state_dim,
    )
