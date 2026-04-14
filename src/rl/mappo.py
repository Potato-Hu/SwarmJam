from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from src.env.swarm_env import SwarmEnv
from src.rl.buffer import RolloutBuffer
from src.rl.policy.actor import MAPPOActor
from src.rl.policy.critic import MAPPOCritic
from src.rl.value_norm import ValueNormalizer
from src.utils.config_loader import get_mappo_config


@dataclass(frozen=True)
class MAPPOConfig:
    rollout_length: int
    batch_size: int
    ppo_epochs: int
    minibatch_size: int
    gamma: float
    gae_lambda: float
    clip_ratio: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float
    learning_rate: float
    use_value_normalization: bool
    value_norm_epsilon: float
    actor_hidden_dim: int
    critic_hidden_dim: int
    actor_num_layers: int
    critic_num_layers: int
    activation: str
    share_policy: bool
    use_centralized_critic: bool
    use_critic_local_observations: bool

    @classmethod
    def from_dict(cls, config: dict[str, Any], num_agents: int) -> "MAPPOConfig":
        training = config.get("training", {})
        network = config.get("network", {})
        marl = config.get("marl", {})

        configured_batch_size = int(training.get("batch_size", 0))
        rollout_length = int(training.get("rollout_length", 32))
        if configured_batch_size > 0:
            if configured_batch_size % num_agents != 0:
                raise ValueError(
                    f"training.batch_size must be divisible by num_agents={num_agents}, got {configured_batch_size}."
                )
            rollout_length = configured_batch_size // num_agents
            batch_size = configured_batch_size
        else:
            batch_size = rollout_length * num_agents

        minibatch_size = int(training.get("minibatch_size", 32))
        if minibatch_size > batch_size:
            raise ValueError(
                f"training.minibatch_size must be <= batch_size, got minibatch_size={minibatch_size}, batch_size={batch_size}."
            )

        return cls(
            rollout_length=rollout_length,
            batch_size=batch_size,
            ppo_epochs=int(training.get("ppo_epochs", 2)),
            minibatch_size=minibatch_size,
            gamma=float(training.get("gamma", 0.99)),
            gae_lambda=float(training.get("gae_lambda", 0.95)),
            clip_ratio=float(training.get("clip_ratio", 0.2)),
            value_loss_coef=float(training.get("value_loss_coef", 0.5)),
            entropy_coef=float(training.get("entropy_coef", 0.01)),
            max_grad_norm=float(training.get("max_grad_norm", 0.5)),
            learning_rate=float(training.get("learning_rate", 3e-4)),
            use_value_normalization=bool(training.get("use_value_normalization", True)),
            value_norm_epsilon=float(training.get("value_norm_epsilon", 1e-4)),
            actor_hidden_dim=int(network.get("actor_hidden_dim", 64)),
            critic_hidden_dim=int(network.get("critic_hidden_dim", 64)),
            actor_num_layers=int(network.get("actor_num_layers", 2)),
            critic_num_layers=int(network.get("critic_num_layers", 2)),
            activation=str(network.get("activation", "relu")),
            share_policy=bool(marl.get("share_policy", True)),
            use_centralized_critic=bool(marl.get("use_centralized_critic", True)),
            use_critic_local_observations=bool(marl.get("use_critic_local_observations", False)),
        )


class MAPPO:
    """Minimal cooperative MAPPO implementation for the debug baseline.

    Engineering support already included for later full-observation integration:
    - value normalization for critic targets
    - PPO epoch/minibatch/clip hyperparameters configurable from YAML
    - batch_size distinct from minibatch_size
    - optional critic input hook for global state + flattened local observations
    """

    def __init__(self, env: SwarmEnv, config: dict[str, Any] | None = None, device: str | torch.device | None = None) -> None:
        if env.space_spec is None:
            raise RuntimeError("Environment spaces are unavailable. Call env.reset() before constructing MAPPO.")

        self.env = env
        self.agent_ids = env.agent_ids
        self.space_spec = env.space_spec
        base_config = get_mappo_config() if config is None else config
        self.config = MAPPOConfig.from_dict(base_config, num_agents=len(self.agent_ids))
        if not self.config.share_policy:
            raise ValueError("This debug implementation currently supports only share_policy=true.")
        if not self.config.use_centralized_critic:
            raise ValueError("This debug implementation requires use_centralized_critic=true.")

        resolved_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)

        self.critic_local_obs_dim = self.space_spec.per_agent_obs_dim * len(self.agent_ids)
        self.actor = MAPPOActor(
            obs_dim=self.space_spec.per_agent_obs_dim,
            action_dim=self.space_spec.per_agent_action_dim,
            hidden_dim=self.config.actor_hidden_dim,
            num_layers=self.config.actor_num_layers,
            activation=self.config.activation,
        ).to(self.device)
        self.critic = MAPPOCritic(
            state_dim=self.space_spec.global_state_dim,
            hidden_dim=self.config.critic_hidden_dim,
            num_layers=self.config.critic_num_layers,
            activation=self.config.activation,
            local_obs_dim=self.critic_local_obs_dim,
            use_local_observations=self.config.use_critic_local_observations,
            num_agents=len(self.agent_ids),
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)
        self.value_normalizer = ValueNormalizer(epsilon=self.config.value_norm_epsilon) if self.config.use_value_normalization else None
        self.buffer = RolloutBuffer(
            rollout_length=self.config.rollout_length,
            num_agents=len(self.agent_ids),
            obs_dim=self.space_spec.per_agent_obs_dim,
            state_dim=self.space_spec.global_state_dim,
            device=self.device,
        )

        self._current_observations: dict[str, np.ndarray] | None = None
        self._current_state: np.ndarray | None = None

    def reset_env(self) -> None:
        observations, _ = self.env.reset()
        self._current_observations = observations
        self._current_state = self.env.get_global_state()

    def get_value_normalizer_state(self) -> dict[str, float] | None:
        return None if self.value_normalizer is None else self.value_normalizer.state_dict()

    def load_value_normalizer_state(self, state: dict[str, float] | None) -> None:
        if self.value_normalizer is not None and state is not None:
            self.value_normalizer.load_state_dict(state)

    def _build_critic_local_observations(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        ordered_obs = np.stack([observations[agent_id] for agent_id in self.agent_ids], axis=0)
        return ordered_obs.reshape(-1).astype(np.float32)

    def _critic_forward(self, state: np.ndarray | torch.Tensor, critic_local_observations: np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
        state_tensor = state if isinstance(state, torch.Tensor) else torch.as_tensor(state, dtype=torch.float32, device=self.device)
        local_obs_tensor: torch.Tensor | None = None
        if self.config.use_critic_local_observations:
            if critic_local_observations is None:
                raise ValueError("critic_local_observations must be provided when use_critic_local_observations=True.")
            local_obs_tensor = (
                critic_local_observations
                if isinstance(critic_local_observations, torch.Tensor)
                else torch.as_tensor(critic_local_observations, dtype=torch.float32, device=self.device)
            )
        return self.critic(state_tensor, local_observations=local_obs_tensor)

    @torch.no_grad()
    def select_actions(
        self,
        observations: dict[str, np.ndarray],
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
        ordered_obs = np.stack([observations[agent_id] for agent_id in self.agent_ids], axis=0)
        obs_tensor = torch.as_tensor(ordered_obs, dtype=torch.float32, device=self.device)
        critic_local_observations = self._build_critic_local_observations(observations)

        actions_tensor, log_probs_tensor = self.actor.act(obs_tensor, deterministic=deterministic)
        value_tensor = self._critic_forward(state, critic_local_observations)
        if self.value_normalizer is not None:
            value_tensor = self.value_normalizer.denormalize(value_tensor)

        actions = {agent_id: int(actions_tensor[idx].item()) for idx, agent_id in enumerate(self.agent_ids)}
        log_probs = log_probs_tensor.detach().cpu().numpy().astype(np.float32)
        values = value_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return actions, log_probs, values, critic_local_observations

    def collect_rollout(self) -> dict[str, float]:
        if self._current_observations is None or self._current_state is None:
            self.reset_env()

        self.buffer.reset()
        episode_return = 0.0
        completed_episodes = 0
        last_done = False

        for _ in range(self.config.rollout_length):
            assert self._current_observations is not None
            assert self._current_state is not None

            stacked_observations = np.stack([self._current_observations[agent_id] for agent_id in self.agent_ids], axis=0)
            actions, log_probs, values, critic_local_observations = self.select_actions(self._current_observations, self._current_state)
            step_output = self.env.step(actions)

            rewards = np.asarray([step_output.rewards[agent_id] for agent_id in self.agent_ids], dtype=np.float32)
            done = bool(
                any(step_output.terminated[agent_id] for agent_id in self.agent_ids)
                or any(step_output.truncated[agent_id] for agent_id in self.agent_ids)
            )
            next_state = self.env.get_global_state()

            self.buffer.add(
                observations=stacked_observations,
                state=self._current_state,
                critic_local_observations=critic_local_observations,
                actions=np.asarray([actions[agent_id] for agent_id in self.agent_ids], dtype=np.int64),
                log_probs=log_probs,
                rewards=rewards,
                done=done,
                values=values,
            )

            episode_return += float(np.mean(rewards))
            last_done = done

            if done:
                completed_episodes += 1
                self.reset_env()
            else:
                self._current_observations = step_output.observations
                self._current_state = next_state

        if last_done:
            last_values = np.zeros(len(self.agent_ids), dtype=np.float32)
        else:
            assert self._current_state is not None
            assert self._current_observations is not None
            with torch.no_grad():
                bootstrap_value = self._critic_forward(
                    self._current_state,
                    self._build_critic_local_observations(self._current_observations),
                )
                if self.value_normalizer is not None:
                    bootstrap_value = self.value_normalizer.denormalize(bootstrap_value)
            last_values = bootstrap_value.squeeze(0).detach().cpu().numpy().astype(np.float32)

        self.buffer.compute_returns_and_advantages(
            last_values=last_values,
            last_done=last_done,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        average_step_reward = float(np.mean(self.buffer.rewards[: self.buffer.position])) if self.buffer.position > 0 else 0.0
        return {
            "rollout_steps": float(self.buffer.position),
            "batch_size": float(self.config.batch_size),
            "completed_episodes": float(completed_episodes),
            "episode_return_sum": float(episode_return),
            "average_step_reward": average_step_reward,
        }

    def update(self) -> dict[str, float]:
        advantages_np = self.buffer.advantages[: self.buffer.position]
        advantage_mean = float(np.mean(advantages_np)) if advantages_np.size > 0 else 0.0
        advantage_std = float(np.std(advantages_np)) if advantages_np.size > 0 else 1.0
        advantage_std = max(advantage_std, 1e-8)

        actor_loss_total = 0.0
        critic_loss_total = 0.0
        entropy_total = 0.0
        update_steps = 0

        for _ in range(self.config.ppo_epochs):
            for batch in self.buffer.iter_minibatches(self.config.minibatch_size):
                normalized_advantages = (batch.advantages - advantage_mean) / advantage_std

                new_log_probs, entropy = self.actor.evaluate_actions(batch.observations, batch.actions)
                ratios = torch.exp(new_log_probs - batch.log_probs)
                unclipped_objective = ratios * normalized_advantages
                clipped_objective = torch.clamp(
                    ratios,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * normalized_advantages
                actor_loss = -torch.min(unclipped_objective, clipped_objective).mean()
                entropy_bonus = entropy.mean()

                predicted_all_values = self._critic_forward(batch.states, batch.critic_local_observations)
                predicted_values = predicted_all_values.gather(1, batch.agent_indices.unsqueeze(1)).squeeze(1)
                if self.value_normalizer is not None:
                    returns_tensor = batch.returns.detach()
                    self.value_normalizer.update(returns_tensor)
                    target_values = self.value_normalizer.normalize(returns_tensor)
                else:
                    target_values = batch.returns
                critic_loss = nn.functional.mse_loss(predicted_values, target_values)

                self.actor_optimizer.zero_grad()
                total_actor_loss = actor_loss - self.config.entropy_coef * entropy_bonus
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_objective = self.config.value_loss_coef * critic_loss
                critic_objective.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

                actor_loss_total += float(actor_loss.item())
                critic_loss_total += float(critic_loss.item())
                entropy_total += float(entropy_bonus.item())
                update_steps += 1

        return {
            "actor_loss": actor_loss_total / max(update_steps, 1),
            "critic_loss": critic_loss_total / max(update_steps, 1),
            "entropy": entropy_total / max(update_steps, 1),
        }
