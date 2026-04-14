from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.entities.enemy_node import EnemyNode
from src.entities.friendly_uav import FriendlyUAV
from src.utils.config_loader import (
    get_default_seed,
    get_enemy_mobility_config,
    get_enemy_vmax,
    get_friendly_vmax,
    should_print_initial_positions,
)

PositionDict = Mapping[str, Sequence[Sequence[float]]]


@dataclass
class ScenarioGroundTruth:
    """Container for scenario initialization, simulation, and plotting."""

    bounds: np.ndarray
    enemy_nodes: list[EnemyNode]
    friendly_uavs: list[FriendlyUAV]
    dt: float = 1.0

    @property
    def key_enemy_nodes(self) -> list[EnemyNode]:
        return [node for node in self.enemy_nodes if node.role == "key"]

    @property
    def non_key_enemy_nodes(self) -> list[EnemyNode]:
        return [node for node in self.enemy_nodes if node.role != "key"]

    def reset(self) -> None:
        for node in self.enemy_nodes:
            node.reset()
        for uav in self.friendly_uavs:
            uav.reset()


def _as_bounds(bounds: Sequence[float] | None) -> np.ndarray:
    mobility_config = get_enemy_mobility_config()
    scene_bounds = np.asarray(bounds if bounds is not None else mobility_config.get("bounds", [1000.0, 1000.0, 300.0]), dtype=float)
    if scene_bounds.shape != (3,):
        raise ValueError(f"bounds must be a 3D vector, got shape {scene_bounds.shape}.")
    return scene_bounds


def _validate_positions(name: str, positions: Sequence[Sequence[float]], expected_count: int, bounds: np.ndarray) -> np.ndarray:
    coords = np.asarray(positions, dtype=float)
    if coords.shape != (expected_count, 3):
        raise ValueError(f"{name} must contain {expected_count} 3D positions, got shape {coords.shape}.")
    if np.any(coords < 0.0) or np.any(coords > bounds):
        raise ValueError(f"{name} positions must stay within bounds {bounds.tolist()}.")
    return coords


def _random_positions(count: int, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if count == 0:
        return np.zeros((0, 3), dtype=float)
    return rng.uniform(low=np.zeros(3, dtype=float), high=bounds, size=(count, 3))


def _nearby_distance_limits(bounds: np.ndarray) -> tuple[float, float]:
    min_distance = float(bounds[0] / 20.0)
    max_distance = float(bounds[0] / 5.0)
    if max_distance <= 0.0:
        raise ValueError("x boundary must be positive to sample nearby initial positions.")
    return min_distance, max_distance


def _sample_near_anchor(
    anchor: np.ndarray,
    bounds: np.ndarray,
    rng: np.random.Generator,
    min_distance: float,
    max_distance: float,
) -> np.ndarray:
    for _ in range(2048):
        direction = rng.normal(size=3)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            continue
        distance = rng.uniform(min_distance, max_distance)
        candidate = anchor + direction / norm * distance
        if np.all(candidate >= 0.0) and np.all(candidate <= bounds):
            return candidate.astype(float)

    candidates = rng.uniform(low=np.zeros(3, dtype=float), high=bounds, size=(8192, 3))
    distances = np.linalg.norm(candidates - anchor, axis=1)
    valid_indices = np.flatnonzero((distances >= min_distance) & (distances <= max_distance))
    if valid_indices.size:
        return candidates[int(rng.choice(valid_indices))].astype(float)

    raise ValueError(
        "Unable to sample a nearby initial position within "
        f"[{min_distance:.3f}, {max_distance:.3f}] and bounds {bounds.tolist()}."
    )


def _anchor_indices_for_nearby_positions(
    count: int,
    anchor_count: int,
    rng: np.random.Generator,
    *,
    entity_name: str,
) -> np.ndarray:
    if count == 0:
        return np.zeros(0, dtype=int)
    if anchor_count == 0:
        raise ValueError(f"Cannot initialize {entity_name} around enemy UAVs because key_num is 0.")
    if count < anchor_count:
        raise ValueError(
            f"{entity_name} count ({count}) must be at least key_num ({anchor_count}) "
            "so every key enemy UAV has one nearby unit."
        )

    required = np.arange(anchor_count, dtype=int)
    extra_count = count - anchor_count
    if extra_count == 0:
        return required
    extras = rng.integers(low=0, high=anchor_count, size=extra_count, dtype=int)
    return np.concatenate([required, extras])


def _sample_near_anchors(
    anchors: np.ndarray,
    count: int,
    bounds: np.ndarray,
    rng: np.random.Generator,
    *,
    entity_name: str,
) -> np.ndarray:
    if count == 0:
        return np.zeros((0, 3), dtype=float)

    min_distance, max_distance = _nearby_distance_limits(bounds)
    anchor_indices = _anchor_indices_for_nearby_positions(
        count,
        len(anchors),
        rng,
        entity_name=entity_name,
    )
    nearby_positions = [
        _sample_near_anchor(
            anchor=anchors[int(anchor_idx)],
            bounds=bounds,
            rng=rng,
            min_distance=min_distance,
            max_distance=max_distance,
        )
        for anchor_idx in anchor_indices
    ]
    return np.asarray(nearby_positions, dtype=float)


def _format_coord(coord: Sequence[float]) -> str:
    vector = np.asarray(coord, dtype=float)
    return f"({vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f})"


def format_scene_positions(
    scene: ScenarioGroundTruth,
    enemy_positions: np.ndarray | None = None,
    friendly_positions: np.ndarray | None = None,
) -> list[str]:
    enemy_coords = enemy_positions if enemy_positions is not None else np.array([node.coords for node in scene.enemy_nodes], dtype=float)
    friendly_coords = friendly_positions if friendly_positions is not None else np.array([uav.coords for uav in scene.friendly_uavs], dtype=float)

    lines: list[str] = []
    for idx, coord in enumerate(enemy_coords):
        lines.append(f"E{idx + 1}:{_format_coord(coord)}")
    for idx, coord in enumerate(friendly_coords):
        lines.append(f"F{idx + 1}:{_format_coord(coord)}")
    return lines


def print_scene_positions(
    scene: ScenarioGroundTruth,
    title: str,
    enemy_positions: np.ndarray | None = None,
    friendly_positions: np.ndarray | None = None,
) -> None:
    print(title)
    for line in format_scene_positions(scene, enemy_positions=enemy_positions, friendly_positions=friendly_positions):
        print(line)


def assign_initial_positions(
    key_num: int,
    nonkey_num: int,
    mydrone_num: int,
    bounds: Sequence[float] | None = None,
    initial_positions: PositionDict | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Resolve initial positions for the scenario.

    `initial_positions` defaults to an empty dict and supports manual coordinates.
    Keys are:
    - `key_enemy`
    - `nonkey_enemy`
    - `friend_drone`

    Each value should be a list of 3D coordinates. Example:
    {
        "key_enemy": [[260.0, 280.0, 150.0], [420.0, 520.0, 150.0]],
        "nonkey_enemy": [[340.0, 400.0, 150.0], [500.0, 360.0, 150.0]],
        "friend_drone": [[250.0, 500.0, 0.0], [500.0, 500.0, 0.0], [750.0, 500.0, 0.0]],
    }
    """

    if key_num < 0 or nonkey_num < 0 or mydrone_num < 0:
        raise ValueError("key_num, nonkey_num, and mydrone_num must be non-negative.")

    scene_bounds = _as_bounds(bounds)
    manual_positions = dict(initial_positions or {})
    position_rng = rng if rng is not None else np.random.default_rng()

    key_positions = (
        _validate_positions("key_enemy", manual_positions["key_enemy"], key_num, scene_bounds)
        if manual_positions.get("key_enemy") is not None
        else _random_positions(key_num, scene_bounds, position_rng)
    )
    nonkey_positions = (
        _validate_positions("nonkey_enemy", manual_positions["nonkey_enemy"], nonkey_num, scene_bounds)
        if manual_positions.get("nonkey_enemy") is not None
        else _sample_near_anchors(
            key_positions,
            nonkey_num,
            scene_bounds,
            position_rng,
            entity_name="nonkey_enemy",
        )
    )
    friendly_positions = (
        _validate_positions("friend_drone", manual_positions["friend_drone"], mydrone_num, scene_bounds)
        if manual_positions.get("friend_drone") is not None
        else _sample_near_anchors(
            key_positions,
            mydrone_num,
            scene_bounds,
            position_rng,
            entity_name="friend_drone",
        )
    )

    enemy_positions = np.vstack([key_positions, nonkey_positions])
    key_indices = np.arange(key_num, dtype=int)

    return {
        "key_enemy": np.asarray(key_positions, dtype=float),
        "nonkey_enemy": np.asarray(nonkey_positions, dtype=float),
        "enemy": enemy_positions,
        "key_indices": key_indices,
        "friend_drone": np.asarray(friendly_positions, dtype=float),
    }


def initialize_scene(
    key_num: int,
    nonkey_num: int,
    mydrone_num: int,
    bounds: Sequence[float] | None = None,
    dt: float = 1.0,
    seed: int | None = None,
    enemy_vmax: float | None = None,
    friendly_vmax: float | None = None,
    initial_positions: PositionDict | None = None,
) -> ScenarioGroundTruth:
    scene_bounds = _as_bounds(bounds)
    base_seed = get_default_seed() if seed is None else int(seed)
    resolved_enemy_vmax = get_enemy_vmax() if enemy_vmax is None else float(enemy_vmax)
    resolved_friendly_vmax = get_friendly_vmax() if friendly_vmax is None else float(friendly_vmax)
    positions = assign_initial_positions(
        key_num=key_num,
        nonkey_num=nonkey_num,
        mydrone_num=mydrone_num,
        bounds=scene_bounds,
        initial_positions=initial_positions,
        rng=np.random.default_rng(base_seed),
    )

    enemy_nodes: list[EnemyNode] = []
    key_indices = set(np.asarray(positions["key_indices"], dtype=int).tolist())
    for node_id, coords in enumerate(positions["enemy"]):
        role = "key" if node_id in key_indices else "non_key"
        enemy_nodes.append(
            EnemyNode(
                node_id=node_id,
                role=role,
                position=coords,
                speed=resolved_enemy_vmax,
                seed=base_seed + node_id,
                direction=0.0 if node_id % 2 == 0 else np.pi,
                pitch=0.04 if role == "key" else -0.04,
            )
        )

    friendly_uavs: list[FriendlyUAV] = []
    for idx, coords in enumerate(positions["friend_drone"]):
        friendly_uavs.append(
            FriendlyUAV(
                node_id=idx,
                position=coords,
                vmax=resolved_friendly_vmax,
                seed=base_seed + 100 + idx,
            )
        )

    scene = ScenarioGroundTruth(
        bounds=scene_bounds,
        enemy_nodes=enemy_nodes,
        friendly_uavs=friendly_uavs,
        dt=float(dt),
    )
    if should_print_initial_positions():
        print_scene_positions(scene, title="Initialization positions:")
    return scene


def create_demo_ground_truth(
    bounds: Sequence[float] | None = None,
    dt: float = 1.0,
    seed: int | None = None,
    enemy_speed: float | None = None,
    friendly_vmax: float | None = None,
) -> ScenarioGroundTruth:
    """Backward-compatible wrapper around `initialize_scene`."""

    scene_bounds = _as_bounds(bounds)
    enemy_y = float(scene_bounds[1] * 0.68)
    enemy_z = float(scene_bounds[2] * 0.58)
    friendly_y = float(scene_bounds[1] * 0.32)
    friendly_z = float(scene_bounds[2] * 0.42)

    return initialize_scene(
        key_num=3,
        nonkey_num=3,
        mydrone_num=3,
        bounds=scene_bounds,
        dt=dt,
        seed=seed,
        enemy_vmax=enemy_speed,
        friendly_vmax=friendly_vmax,
        initial_positions={
            "key_enemy": [
                [scene_bounds[0] * 0.18, enemy_y, enemy_z],
                [scene_bounds[0] * 0.436, enemy_y, enemy_z],
                [scene_bounds[0] * 0.692, enemy_y, enemy_z],
            ],
            "nonkey_enemy": [
                [scene_bounds[0] * 0.308, enemy_y, enemy_z],
                [scene_bounds[0] * 0.564, enemy_y, enemy_z],
                [scene_bounds[0] * 0.82, enemy_y, enemy_z],
            ],
            "friend_drone": [
                [scene_bounds[0] * 0.28, friendly_y, friendly_z],
                [scene_bounds[0] * 0.5, friendly_y, friendly_z],
                [scene_bounds[0] * 0.72, friendly_y, friendly_z],
            ],
        },
    )
