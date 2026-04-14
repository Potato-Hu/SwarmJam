from __future__ import annotations

__all__ = [
    "GaussMarkov3DMobility",
    "RandomWalk3DMobility",
    "ScenarioGroundTruth",
    "TrajectoryBundle",
    "assign_initial_positions",
    "build_enemy_mobility",
    "create_demo_ground_truth",
    "generate_demo_trajectories",
    "generate_enemy_trajectory",
    "generate_friendly_trajectory",
    "generate_scene_trajectories",
    "initialize_scene",
]


def __getattr__(name: str):
    if name in {"GaussMarkov3DMobility", "RandomWalk3DMobility", "build_enemy_mobility"}:
        from src.simulation.enemy_mobility import (
            GaussMarkov3DMobility,
            RandomWalk3DMobility,
            build_enemy_mobility,
        )

        exports = {
            "GaussMarkov3DMobility": GaussMarkov3DMobility,
            "RandomWalk3DMobility": RandomWalk3DMobility,
            "build_enemy_mobility": build_enemy_mobility,
        }
        return exports[name]

    if name in {"ScenarioGroundTruth", "assign_initial_positions", "create_demo_ground_truth", "initialize_scene"}:
        from src.simulation.ground_truth import (
            ScenarioGroundTruth,
            assign_initial_positions,
            create_demo_ground_truth,
            initialize_scene,
        )

        exports = {
            "ScenarioGroundTruth": ScenarioGroundTruth,
            "assign_initial_positions": assign_initial_positions,
            "create_demo_ground_truth": create_demo_ground_truth,
            "initialize_scene": initialize_scene,
        }
        return exports[name]

    if name in {"TrajectoryBundle", "generate_demo_trajectories", "generate_enemy_trajectory", "generate_friendly_trajectory", "generate_scene_trajectories"}:
        from src.simulation.trajectory_generator import (
            TrajectoryBundle,
            generate_demo_trajectories,
            generate_enemy_trajectory,
            generate_friendly_trajectory,
            generate_scene_trajectories,
        )

        exports = {
            "TrajectoryBundle": TrajectoryBundle,
            "generate_demo_trajectories": generate_demo_trajectories,
            "generate_enemy_trajectory": generate_enemy_trajectory,
            "generate_friendly_trajectory": generate_friendly_trajectory,
            "generate_scene_trajectories": generate_scene_trajectories,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
