from __future__ import annotations

__all__ = [
    "generate_demo_figures",
    "plot_scene_snapshot",
    "plot_trajectories_until_timestep",
]


def __getattr__(name: str):
    if name in __all__:
        from src.visualization.plot_scene import (
            generate_demo_figures,
            plot_scene_snapshot,
            plot_trajectories_until_timestep,
        )

        exports = {
            "generate_demo_figures": generate_demo_figures,
            "plot_scene_snapshot": plot_scene_snapshot,
            "plot_trajectories_until_timestep": plot_trajectories_until_timestep,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
