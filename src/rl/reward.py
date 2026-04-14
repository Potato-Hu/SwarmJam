"""Reward utilities for multi-UAV cooperative tracking and jamming."""

from __future__ import annotations

import numpy as np


def _global_jamming_utility(target_jamming_powers: np.ndarray, J0: float) -> float:
    """Normalized target-jamming utility with diminishing returns."""

    return float(np.sum(np.log1p(np.asarray(target_jamming_powers, dtype=float) / float(J0))))


def compute_difference_rewards(
    pairwise_jamming_powers: np.ndarray,
    ally_interference_powers: np.ndarray,
    tau_ally: float,
    J0: float,
    lambda_safety: float,
) -> np.ndarray:
    """Compute pure per-agent log-power difference rewards."""

    if J0 <= 0:
        raise ValueError(f"J0 must be positive, got {J0}.")

    pairwise_jamming_powers = np.asarray(pairwise_jamming_powers, dtype=float)
    ally_interference_powers = np.asarray(ally_interference_powers, dtype=float).reshape(-1)
    if pairwise_jamming_powers.ndim != 2:
        raise ValueError(f"pairwise_jamming_powers must be 2D, got shape {pairwise_jamming_powers.shape}.")

    num_friendlies, num_targets = pairwise_jamming_powers.shape
    if num_targets == 0:
        raise ValueError("pairwise_jamming_powers must contain at least one target.")
    if ally_interference_powers.size == 0:
        ally_interference_powers = np.zeros(num_friendlies, dtype=float)
    if ally_interference_powers.shape != (num_friendlies,):
        raise ValueError(
            f"ally_interference_powers must have shape ({num_friendlies},), got {ally_interference_powers.shape}."
        )

    total_target_powers = pairwise_jamming_powers.sum(axis=0)
    global_utility = _global_jamming_utility(total_target_powers, J0=J0) / float(num_targets)

    rewards = np.zeros(num_friendlies, dtype=float)
    for friendly_idx in range(num_friendlies):
        counterfactual_powers = total_target_powers - pairwise_jamming_powers[friendly_idx]
        counterfactual_utility = _global_jamming_utility(counterfactual_powers, J0=J0) / float(num_targets)
        difference_reward = global_utility - counterfactual_utility
        safety_penalty = max(0.0, ally_interference_powers[friendly_idx] - float(tau_ally))
        rewards[friendly_idx] = difference_reward - float(lambda_safety) * safety_penalty

    return rewards


def compute_distance_assignment(
    pairwise_distances: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assign key targets dynamically to friendly UAVs.

    Each key target is first assigned to the nearest still-unassigned friendly UAV.
    If there are more friendly UAVs than key targets, remaining friendlies are assigned
    to their nearest key target. This keeps all key targets covered when enough
    friendlies are available, while allowing surplus friendlies to reinforce the nearest
    target instead of staying idle.
    """

    pairwise_distances = np.asarray(pairwise_distances, dtype=float)
    if pairwise_distances.ndim != 2:
        raise ValueError(f"pairwise_distances must be 2D, got shape {pairwise_distances.shape}.")

    num_friendlies, num_targets = pairwise_distances.shape
    if num_targets == 0:
        raise ValueError("pairwise_distances must contain at least one key target.")

    assigned_targets = np.full(num_friendlies, -1, dtype=int)
    assigned_distances = np.full(num_friendlies, np.inf, dtype=float)
    unassigned_friendlies = set(range(num_friendlies))

    # Priority 1: cover each key target with its nearest still-unassigned friendly UAV.
    # This avoids the global closest-pair rule consuming several friendlies near one target
    # before every key target has a responsible tracker.
    for target_idx in range(num_targets):
        if not unassigned_friendlies:
            break
        friendly_idx = min(
            unassigned_friendlies,
            key=lambda candidate: float(pairwise_distances[candidate, target_idx]),
        )
        assigned_targets[friendly_idx] = target_idx
        assigned_distances[friendly_idx] = float(pairwise_distances[friendly_idx, target_idx])
        unassigned_friendlies.remove(friendly_idx)

    # Priority 2: surplus friendlies reinforce the nearest key target to themselves.
    for friendly_idx in sorted(unassigned_friendlies):
        target_idx = int(np.argmin(pairwise_distances[friendly_idx]))
        assigned_targets[friendly_idx] = target_idx
        assigned_distances[friendly_idx] = float(pairwise_distances[friendly_idx, target_idx])

    return {
        "assigned_target": assigned_targets.astype(float),
        "assigned_distance": assigned_distances,
    }


def compute_power_progress_rewards(
    assigned_targets: np.ndarray,
    assigned_distances: np.ndarray,
    previous_assigned_targets: np.ndarray,
    previous_assigned_distances: np.ndarray,
    pairwise_jamming_powers: np.ndarray,
    ally_interference_powers: np.ndarray,
    actions: np.ndarray,
    *,
    J0: float,
    tau_ally: float,
    lambda_safety: float,
    power_weight: float,
    progress_weight: float,
    progress_distance_scale: float,
    move_penalty_weight: float,
    stop_action: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute power and distance-progress rewards for assigned key targets."""

    if J0 <= 0:
        raise ValueError(f"J0 must be positive, got {J0}.")
    if progress_distance_scale <= 0:
        raise ValueError(f"progress_distance_scale must be positive, got {progress_distance_scale}.")

    assigned_targets = np.asarray(assigned_targets, dtype=int).reshape(-1)
    assigned_distances = np.asarray(assigned_distances, dtype=float).reshape(-1)
    previous_assigned_targets = np.asarray(previous_assigned_targets, dtype=int).reshape(-1)
    previous_assigned_distances = np.asarray(previous_assigned_distances, dtype=float).reshape(-1)
    pairwise_jamming_powers = np.asarray(pairwise_jamming_powers, dtype=float)
    ally_interference_powers = np.asarray(ally_interference_powers, dtype=float).reshape(-1)
    actions = np.asarray(actions, dtype=int).reshape(-1)

    if pairwise_jamming_powers.ndim != 2:
        raise ValueError(f"pairwise_jamming_powers must be 2D, got shape {pairwise_jamming_powers.shape}.")
    num_friendlies, num_targets = pairwise_jamming_powers.shape
    if num_targets == 0:
        raise ValueError("pairwise_jamming_powers must contain at least one target.")

    expected_shape = (num_friendlies,)
    for name, array in {
        "assigned_targets": assigned_targets,
        "assigned_distances": assigned_distances,
        "previous_assigned_targets": previous_assigned_targets,
        "previous_assigned_distances": previous_assigned_distances,
        "actions": actions,
    }.items():
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}.")

    if ally_interference_powers.size == 0:
        ally_interference_powers = np.zeros(num_friendlies, dtype=float)
    if ally_interference_powers.shape != expected_shape:
        raise ValueError(
            f"ally_interference_powers must have shape {expected_shape}, got {ally_interference_powers.shape}."
        )
    if np.any((assigned_targets < 0) | (assigned_targets >= num_targets)):
        raise ValueError(f"assigned_targets must be in [0, {num_targets}), got {assigned_targets}.")

    friendly_indices = np.arange(num_friendlies)
    assigned_powers = pairwise_jamming_powers[friendly_indices, assigned_targets]
    power_reward = float(power_weight) * np.log1p(assigned_powers / float(J0))

    same_target = assigned_targets == previous_assigned_targets
    valid_prev = np.isfinite(previous_assigned_distances) & same_target
    distance_delta = np.zeros_like(assigned_distances, dtype=float)
    distance_delta[valid_prev] = previous_assigned_distances[valid_prev] - assigned_distances[valid_prev]
    progress_term = np.clip(distance_delta / float(progress_distance_scale), -1.0, 1.0)
    progress_reward = float(progress_weight) * progress_term

    move_penalty = np.where(actions == int(stop_action), 0.0, 1.0)
    safety_penalty = np.maximum(0.0, ally_interference_powers - float(tau_ally))
    rewards = (
        power_reward
        + progress_reward
        - float(move_penalty_weight) * move_penalty
        - float(lambda_safety) * safety_penalty
    )

    components = {
        "assigned_target": assigned_targets.astype(float),
        "assigned_distance": assigned_distances.astype(float),
        "assigned_power": assigned_powers.astype(float),
        "power_reward": power_reward.astype(float),
        "progress_reward": progress_reward.astype(float),
        "move_penalty": move_penalty.astype(float),
        "safety_penalty": safety_penalty.astype(float),
    }
    return rewards.astype(float), components
