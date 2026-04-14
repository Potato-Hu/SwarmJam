from __future__ import annotations

import torch
from torch import nn

from src.rl.policy.feature_encoder import FeatureEncoder


class MAPPOCritic(nn.Module):
    """Centralized critic that estimates one state value per friendly agent.

    The input can optionally concatenate global state and flattened local observations.
    For the current debug baseline, `local_observations` is left unused and the critic
    sees only the global truth state. This hook is kept so we can later plug in
    association/sensing outputs without rewriting the critic backbone.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "relu",
        local_obs_dim: int = 0,
        use_local_observations: bool = False,
        num_agents: int = 1,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.local_obs_dim = int(local_obs_dim)
        self.use_local_observations = bool(use_local_observations)
        self.num_agents = int(num_agents)
        self.input_dim = self.state_dim + (self.local_obs_dim if self.use_local_observations else 0)
        self.encoder = FeatureEncoder(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
        )
        self.value_head = nn.Linear(self.encoder.output_dim, self.num_agents)

    def forward(self, states: torch.Tensor, local_observations: torch.Tensor | None = None) -> torch.Tensor:
        if states.dim() == 1:
            states = states.unsqueeze(0)

        critic_inputs = states
        if self.use_local_observations:
            if local_observations is None:
                raise ValueError("local_observations must be provided when use_local_observations=True.")
            if local_observations.dim() == 1:
                local_observations = local_observations.unsqueeze(0)
            critic_inputs = torch.cat([states, local_observations], dim=-1)

        features = self.encoder(critic_inputs)
        return self.value_head(features)
