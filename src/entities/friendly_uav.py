from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np

from src.utils.config_loader import get_default_seed
from src.utils.seed import make_rng


def _build_action_directions() -> tuple[np.ndarray, ...]:
    directions: list[np.ndarray] = []
    for x in (0.0, 1.0, -1.0):
        for y in (0.0, 1.0, -1.0):
            for z in (0.0, 1.0, -1.0):
                vector = np.array([x, y, z], dtype=float)
                norm = float(np.linalg.norm(vector))
                directions.append(vector if norm == 0.0 else vector / norm)
    return tuple(directions)


def _action_name_from_direction(direction: np.ndarray) -> str:
    if np.allclose(direction, 0.0):
        return "stop"

    parts: list[str] = []
    for axis_name, value in zip(("x", "y", "z"), direction, strict=True):
        if value > 0.0:
            parts.append(f"+{axis_name}")
        elif value < 0.0:
            parts.append(f"-{axis_name}")
    return "".join(parts)


def _as_vector3(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3D vector, got shape {array.shape}.")
    return array


@dataclass
class FriendlyUAV:
    """Friendly UAV with a compact discrete action space."""

    node_id: int
    position: Sequence[float]
    vmax: float
    seed: int | None = None
    action: int = 0

    rng: np.random.Generator = field(init=False, repr=False)
    coords: np.ndarray = field(init=False, repr=False)
    velocity_vector: np.ndarray = field(init=False, repr=False)
    speed: float = field(init=False)
    initial_coords: np.ndarray = field(init=False, repr=False)
    initial_action: int = field(init=False)

    ACTION_DIRECTIONS: ClassVar[tuple[np.ndarray, ...]] = _build_action_directions()
    ACTION_NAMES: ClassVar[tuple[str, ...]] = tuple(_action_name_from_direction(direction) for direction in ACTION_DIRECTIONS)

    def __post_init__(self) -> None:
        if self.vmax < 0.0:
            raise ValueError("vmax must be non-negative.")

        self.rng = make_rng(self.seed, offset=self.node_id, fallback=get_default_seed())
        self.coords = _as_vector3(self.position, name="position")
        self.initial_coords = self.coords.copy()
        self.initial_action = int(self.action)

        self._validate_action(self.action)
        self.velocity_vector = self.action_to_velocity(self.action)
        self.speed = float(np.linalg.norm(self.velocity_vector))

    @property
    def action_dim(self) -> int:
        return len(self.ACTION_DIRECTIONS)

    @property
    def x(self) -> float:
        return float(self.coords[0])

    @property
    def y(self) -> float:
        return float(self.coords[1])

    @property
    def z(self) -> float:
        return float(self.coords[2])

    @property
    def vx(self) -> float:
        return float(self.velocity_vector[0])

    @property
    def vy(self) -> float:
        return float(self.velocity_vector[1])

    @property
    def vz(self) -> float:
        return float(self.velocity_vector[2])

    def reset(self) -> None:
        self.coords = self.initial_coords.copy()
        self.action = self.initial_action
        self.velocity_vector = self.action_to_velocity(self.action)
        self.speed = float(np.linalg.norm(self.velocity_vector))

    def set_position(self, position: Sequence[float]) -> None:
        self.coords = _as_vector3(position, name="position")

    def action_to_velocity(self, action: int) -> np.ndarray:
        self._validate_action(action)
        direction = self.ACTION_DIRECTIONS[int(action)]
        return direction * float(self.vmax)

    def set_action(self, action: int) -> np.ndarray:
        self._validate_action(action)
        self.action = int(action)
        self.velocity_vector = self.action_to_velocity(self.action)
        self.speed = float(np.linalg.norm(self.velocity_vector))
        return self.velocity_vector.copy()

    def update(self, dt: float, action: int | None = None) -> np.ndarray:
        """Apply a discrete action and advance one step with fixed speed."""

        if dt < 0.0:
            raise ValueError("dt must be non-negative.")
        if action is not None:
            self.set_action(action)

        self.coords = self.coords + self.velocity_vector * float(dt)
        return self.coords.copy()

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "coords": self.coords.copy(),
            "vmax": self.vmax,
            "action": self.action,
            "speed": self.speed,
            "velocity": self.velocity_vector.copy(),
            "seed": self.seed if self.seed is not None else get_default_seed(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FriendlyUAV":
        return cls(
            node_id=int(data["node_id"]),
            position=data["coords"],
            vmax=float(data["vmax"]),
            seed=data.get("seed"),
            action=int(data.get("action", 0)),
        )

    @classmethod
    def action_name(cls, action: int) -> str:
        cls._validate_action(action)
        return cls.ACTION_NAMES[int(action)]

    @classmethod
    def _validate_action(cls, action: int) -> None:
        if int(action) < 0 or int(action) >= len(cls.ACTION_DIRECTIONS):
            raise ValueError(
                f"action must be in [0, {len(cls.ACTION_DIRECTIONS) - 1}], got {action}."
            )
