from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.env.world import SwarmWorld
from src.simulation.ground_truth import ScenarioGroundTruth, initialize_scene, print_scene_positions
from src.utils.config_loader import get_num_steps, should_print_timestep_positions


@dataclass
class TrajectoryBundle:
    enemy_positions: np.ndarray
    friendly_positions: np.ndarray
    friendly_actions: np.ndarray
    key_enemy_interference_watts: np.ndarray
    key_enemy_interference_dbm: np.ndarray
    friendly_interference_watts: np.ndarray
    friendly_interference_dbm: np.ndarray
    dt: float
    bounds: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(self.enemy_positions.shape[0] - 1)


def _print_timestep_positions_from_bundle(scene: ScenarioGroundTruth, bundle: TrajectoryBundle) -> None:
    if not should_print_timestep_positions():
        return

    for step_idx in range(1, bundle.num_steps + 1):
        print_scene_positions(
            scene,
            title=f"Timestep {step_idx} positions:",
            enemy_positions=bundle.enemy_positions[step_idx],
            friendly_positions=bundle.friendly_positions[step_idx],
        )


def _rollout_world(
    scene: ScenarioGroundTruth,
    num_steps: int,
    dt: float | None = None,
    friendly_actions_sequence: Sequence[Sequence[int]] | None = None,
) -> TrajectoryBundle:
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")

    world = SwarmWorld.from_scene(scene)
    step_dt = float(scene.dt if dt is None else dt)
    world.dt = step_dt
    world.reset()

    num_key_enemies = len(world.get_key_enemy_nodes())
    num_friendlies = len(world.friendly_uavs)

    enemy_positions = np.zeros((num_steps + 1, len(world.enemy_nodes), 3), dtype=float)
    friendly_positions = np.zeros((num_steps + 1, num_friendlies, 3), dtype=float)
    friendly_actions = np.zeros((num_steps, num_friendlies), dtype=int)
    key_enemy_interference_watts = np.zeros((num_steps + 1, num_key_enemies), dtype=float)
    key_enemy_interference_dbm = np.zeros((num_steps + 1, num_key_enemies), dtype=float)
    friendly_interference_watts = np.zeros((num_steps + 1, num_friendlies), dtype=float)
    friendly_interference_dbm = np.zeros((num_steps + 1, num_friendlies), dtype=float)

    enemy_positions[0] = world.get_enemy_positions()
    friendly_positions[0] = world.get_friendly_positions()
    key_enemy_interference_watts[0] = world.interference.key_enemy_received_watts
    key_enemy_interference_dbm[0] = world.interference.key_enemy_received_dbm
    friendly_interference_watts[0] = world.interference.friendly_received_watts
    friendly_interference_dbm[0] = world.interference.friendly_received_dbm

    for step_idx in range(1, num_steps + 1):
        step_actions = None if friendly_actions_sequence is None else friendly_actions_sequence[step_idx - 1]
        result = world.step(friendly_actions=step_actions, dt=step_dt)
        enemy_positions[step_idx] = result.enemy_positions
        friendly_positions[step_idx] = result.friendly_positions
        friendly_actions[step_idx - 1] = result.friendly_actions
        key_enemy_interference_watts[step_idx] = result.key_enemy_interference_watts
        key_enemy_interference_dbm[step_idx] = result.key_enemy_interference_dbm
        friendly_interference_watts[step_idx] = result.friendly_interference_watts
        friendly_interference_dbm[step_idx] = result.friendly_interference_dbm

    return TrajectoryBundle(
        enemy_positions=enemy_positions,
        friendly_positions=friendly_positions,
        friendly_actions=friendly_actions,
        key_enemy_interference_watts=key_enemy_interference_watts,
        key_enemy_interference_dbm=key_enemy_interference_dbm,
        friendly_interference_watts=friendly_interference_watts,
        friendly_interference_dbm=friendly_interference_dbm,
        dt=step_dt,
        bounds=world.bounds.copy(),
    )


def generate_enemy_trajectory(
    scene: ScenarioGroundTruth,
    num_steps: int,
    dt: float | None = None,
) -> np.ndarray:
    hold_actions = np.zeros((num_steps, len(scene.friendly_uavs)), dtype=int)
    return _rollout_world(
        scene,
        num_steps=num_steps,
        dt=dt,
        friendly_actions_sequence=hold_actions,
    ).enemy_positions


def generate_friendly_trajectory(
    scene: ScenarioGroundTruth,
    num_steps: int,
    dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    bundle = _rollout_world(scene, num_steps=num_steps, dt=dt)
    return bundle.friendly_positions, bundle.friendly_actions


def generate_scene_trajectories(
    num_steps: int | None = None,
    dt: float = 1.0,
    key_num: int = 3,
    nonkey_num: int = 3,
    mydrone_num: int = 3,
    bounds: Sequence[float] | None = None,
    seed: int | None = None,
    enemy_vmax: float | None = None,
    friendly_vmax: float | None = None,
    initial_positions: Mapping[str, Sequence[Sequence[float]]] | None = None,
) -> tuple[ScenarioGroundTruth, TrajectoryBundle]:
    resolved_num_steps = get_num_steps() if num_steps is None else int(num_steps)
    scene = initialize_scene(
        key_num=key_num,
        nonkey_num=nonkey_num,
        mydrone_num=mydrone_num,
        bounds=bounds,
        dt=dt,
        seed=seed,
        enemy_vmax=enemy_vmax,
        friendly_vmax=friendly_vmax,
        initial_positions=initial_positions,
    )

    bundle = _rollout_world(scene, num_steps=resolved_num_steps, dt=dt)
    _print_timestep_positions_from_bundle(scene, bundle)
    return scene, bundle


def generate_demo_trajectories(
    num_steps: int | None = None,
    dt: float = 1.0,
    bounds: Sequence[float] | None = None,
    seed: int | None = None,
    enemy_speed: float | None = None,
    friendly_vmax: float | None = None,
) -> tuple[ScenarioGroundTruth, TrajectoryBundle]:
    return generate_scene_trajectories(
        num_steps=num_steps,
        dt=dt,
        key_num=3,
        nonkey_num=3,
        mydrone_num=3,
        bounds=bounds,
        seed=seed,
        enemy_vmax=enemy_speed,
        friendly_vmax=friendly_vmax,
    )
