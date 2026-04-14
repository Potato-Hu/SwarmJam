from __future__ import annotations

import numpy as np

from src.rl.mappo import MAPPO
from src.simulation.ground_truth import ScenarioGroundTruth
from src.simulation.trajectory_generator import TrajectoryBundle


def evaluate_policy(trainer: MAPPO, num_episodes: int = 3) -> dict[str, float]:
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    for _ in range(num_episodes):
        _, _, episode_stats = rollout_policy_episode(trainer, deterministic=True)
        episode_returns.append(float(episode_stats["episode_return"]))
        episode_lengths.append(int(episode_stats["episode_length"]))

    return {
        "eval_avg_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "eval_std_return": float(np.std(episode_returns)) if episode_returns else 0.0,
        "eval_avg_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
    }



def rollout_policy_episode(
    trainer: MAPPO,
    deterministic: bool = True,
) -> tuple[ScenarioGroundTruth, TrajectoryBundle, dict[str, float]]:
    trainer.reset_env()
    assert trainer._current_observations is not None
    assert trainer._current_state is not None
    assert trainer.env.world is not None
    assert trainer.env.scene is not None

    world = trainer.env.world
    scene = trainer.env.scene
    max_steps = trainer.env.max_steps
    num_enemies = len(world.enemy_nodes)
    num_friendlies = len(world.friendly_uavs)
    num_key_enemies = len(world.get_key_enemy_nodes())

    enemy_positions = np.zeros((max_steps + 1, num_enemies, 3), dtype=float)
    friendly_positions = np.zeros((max_steps + 1, num_friendlies, 3), dtype=float)
    friendly_actions = np.zeros((max_steps, num_friendlies), dtype=int)
    key_enemy_interference_watts = np.zeros((max_steps + 1, num_key_enemies), dtype=float)
    key_enemy_interference_dbm = np.zeros((max_steps + 1, num_key_enemies), dtype=float)
    friendly_interference_watts = np.zeros((max_steps + 1, num_friendlies), dtype=float)
    friendly_interference_dbm = np.zeros((max_steps + 1, num_friendlies), dtype=float)

    enemy_positions[0] = world.get_enemy_positions()
    friendly_positions[0] = world.get_friendly_positions()
    key_enemy_interference_watts[0] = world.interference.key_enemy_received_watts
    key_enemy_interference_dbm[0] = world.interference.key_enemy_received_dbm
    friendly_interference_watts[0] = world.interference.friendly_received_watts
    friendly_interference_dbm[0] = world.interference.friendly_received_dbm

    episode_return = 0.0
    episode_length = 0
    done = False

    while not done and episode_length < max_steps:
        actions, _, _, _ = trainer.select_actions(
            trainer._current_observations,
            trainer._current_state,
            deterministic=deterministic,
        )
        step_output = trainer.env.step(actions)
        episode_length += 1

        assert trainer.env.world is not None
        world = trainer.env.world
        enemy_positions[episode_length] = world.get_enemy_positions()
        friendly_positions[episode_length] = world.get_friendly_positions()
        friendly_actions[episode_length - 1] = np.asarray(
            [actions[agent_id] for agent_id in trainer.agent_ids],
            dtype=int,
        )
        key_enemy_interference_watts[episode_length] = world.interference.key_enemy_received_watts
        key_enemy_interference_dbm[episode_length] = world.interference.key_enemy_received_dbm
        friendly_interference_watts[episode_length] = world.interference.friendly_received_watts
        friendly_interference_dbm[episode_length] = world.interference.friendly_received_dbm

        rewards = np.asarray([step_output.rewards[agent_id] for agent_id in trainer.agent_ids], dtype=float)
        episode_return += float(np.mean(rewards))
        done = bool(
            any(step_output.terminated[agent_id] for agent_id in trainer.agent_ids)
            or any(step_output.truncated[agent_id] for agent_id in trainer.agent_ids)
        )

        if not done:
            trainer._current_observations = step_output.observations
            trainer._current_state = trainer.env.get_global_state()

    bundle = TrajectoryBundle(
        enemy_positions=enemy_positions[: episode_length + 1],
        friendly_positions=friendly_positions[: episode_length + 1],
        friendly_actions=friendly_actions[:episode_length],
        key_enemy_interference_watts=key_enemy_interference_watts[: episode_length + 1],
        key_enemy_interference_dbm=key_enemy_interference_dbm[: episode_length + 1],
        friendly_interference_watts=friendly_interference_watts[: episode_length + 1],
        friendly_interference_dbm=friendly_interference_dbm[: episode_length + 1],
        dt=trainer.env.dt,
        bounds=world.bounds.copy(),
    )
    stats = {
        "episode_return": float(episode_return),
        "episode_length": float(episode_length),
    }
    return scene, bundle, stats
