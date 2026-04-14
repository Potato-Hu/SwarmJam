from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from src.rl.policy.feature_encoder import FeatureEncoder


class MAPPOActor(nn.Module):
    """Shared discrete actor used by all friendly UAV agents."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.encoder = FeatureEncoder(
            input_dim=self.obs_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
        )
        self.policy_head = nn.Linear(self.encoder.output_dim, self.action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        features = self.encoder(observations)
        return self.policy_head(features)

    def get_dist(self, observations: torch.Tensor) -> Categorical:
        logits = self.forward(observations)
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, observations: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_dist(observations)
        if deterministic:
            actions = torch.argmax(dist.logits, dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_dist(observations)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy
