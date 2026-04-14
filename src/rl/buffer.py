from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RolloutBatch:
    observations: torch.Tensor
    states: torch.Tensor
    critic_local_observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    agent_indices: torch.Tensor


class RolloutBuffer:
    """Rollout storage for the simplified cooperative MAPPO debug pipeline.

    `batch_size` is the total number of agent-time samples used per PPO update.
    In this single-environment implementation we realize that batch by collecting
    `rollout_length = batch_size / num_agents` environment steps, then flattening the
    agent dimension before minibatch sampling.
    """

    def __init__(self, rollout_length: int, num_agents: int, obs_dim: int, state_dim: int, device: torch.device) -> None:
        self.rollout_length = int(rollout_length)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.device = device

        self.observations = np.zeros((self.rollout_length, self.num_agents, self.obs_dim), dtype=np.float32)
        self.states = np.zeros((self.rollout_length, self.state_dim), dtype=np.float32)
        self.critic_local_observations = np.zeros((self.rollout_length, self.num_agents * self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.rollout_length, self.num_agents), dtype=np.int64)
        self.log_probs = np.zeros((self.rollout_length, self.num_agents), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_length, self.num_agents), dtype=np.float32)
        self.dones = np.zeros(self.rollout_length, dtype=np.float32)
        self.values = np.zeros((self.rollout_length, self.num_agents), dtype=np.float32)
        self.returns = np.zeros((self.rollout_length, self.num_agents), dtype=np.float32)
        self.advantages = np.zeros((self.rollout_length, self.num_agents), dtype=np.float32)
        self.position = 0

    def reset(self) -> None:
        self.position = 0

    def add(
        self,
        observations: np.ndarray,
        state: np.ndarray,
        critic_local_observations: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        done: bool,
        values: np.ndarray,
    ) -> None:
        if self.position >= self.rollout_length:
            raise RuntimeError("RolloutBuffer is full. Call reset() before adding more samples.")

        index = self.position
        self.observations[index] = observations
        self.states[index] = state
        self.critic_local_observations[index] = critic_local_observations
        self.actions[index] = actions
        self.log_probs[index] = log_probs
        rewards_array = np.asarray(rewards, dtype=np.float32)
        if rewards_array.shape != (self.num_agents,):
            raise ValueError(f"rewards must have shape ({self.num_agents},), got {rewards_array.shape}.")
        self.rewards[index] = rewards_array
        self.dones[index] = float(done)
        values_array = np.asarray(values, dtype=np.float32)
        if values_array.shape != (self.num_agents,):
            raise ValueError(f"values must have shape ({self.num_agents},), got {values_array.shape}.")
        self.values[index] = values_array
        self.position += 1

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        last_done: bool,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        gae = np.zeros(self.num_agents, dtype=np.float32)
        last_values_array = np.asarray(last_values, dtype=np.float32)
        if last_values_array.shape != (self.num_agents,):
            raise ValueError(f"last_values must have shape ({self.num_agents},), got {last_values_array.shape}.")

        for step in range(self.position - 1, -1, -1):
            if step == self.position - 1:
                next_value = last_values_array
                next_non_terminal = 1.0 - float(last_done)
            else:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]

            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]

    def iter_minibatches(self, minibatch_size: int) -> list[RolloutBatch]:
        rollout_steps = self.position
        total_samples = rollout_steps * self.num_agents
        if total_samples == 0:
            return []

        repeated_advantages = self.advantages[:rollout_steps].reshape(total_samples)
        repeated_returns = self.returns[:rollout_steps].reshape(total_samples)
        repeated_values = self.values[:rollout_steps].reshape(total_samples)
        repeated_states = np.repeat(self.states[:rollout_steps], self.num_agents, axis=0)
        repeated_critic_local_observations = np.repeat(self.critic_local_observations[:rollout_steps], self.num_agents, axis=0)
        repeated_agent_indices = np.tile(np.arange(self.num_agents, dtype=np.int64), rollout_steps)

        observations = torch.as_tensor(
            self.observations[:rollout_steps].reshape(total_samples, self.obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        states = torch.as_tensor(repeated_states, dtype=torch.float32, device=self.device)
        critic_local_observations = torch.as_tensor(
            repeated_critic_local_observations,
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            self.actions[:rollout_steps].reshape(total_samples),
            dtype=torch.long,
            device=self.device,
        )
        log_probs = torch.as_tensor(
            self.log_probs[:rollout_steps].reshape(total_samples),
            dtype=torch.float32,
            device=self.device,
        )
        returns = torch.as_tensor(repeated_returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(repeated_advantages, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(repeated_values, dtype=torch.float32, device=self.device)
        agent_indices = torch.as_tensor(repeated_agent_indices, dtype=torch.long, device=self.device)

        permutation = torch.randperm(total_samples, device=self.device)
        minibatches: list[RolloutBatch] = []
        for start in range(0, total_samples, minibatch_size):
            batch_indices = permutation[start : start + minibatch_size]
            minibatches.append(
                RolloutBatch(
                    observations=observations[batch_indices],
                    states=states[batch_indices],
                    critic_local_observations=critic_local_observations[batch_indices],
                    actions=actions[batch_indices],
                    log_probs=log_probs[batch_indices],
                    returns=returns[batch_indices],
                    advantages=advantages[batch_indices],
                    values=values[batch_indices],
                    agent_indices=agent_indices[batch_indices],
                )
            )
        return minibatches
