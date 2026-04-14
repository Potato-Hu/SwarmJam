from __future__ import annotations

import torch
from torch import nn


def get_activation(name: str) -> type[nn.Module]:
    normalized = name.strip().lower()
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
    }
    if normalized not in activations:
        raise ValueError(f"Unsupported activation '{name}'. Available: {sorted(activations)}")
    return activations[normalized]


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    activation: str = "relu",
    output_dim: int | None = None,
) -> nn.Sequential:
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}.")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}.")

    activation_cls = get_activation(activation)
    dims: list[int] = [input_dim] + [hidden_dim] * num_layers
    modules: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=False):
        modules.append(nn.Linear(in_dim, out_dim))
        modules.append(activation_cls())

    if output_dim is not None:
        modules.append(nn.Linear(dims[-1], output_dim))

    return nn.Sequential(*modules)


class FeatureEncoder(nn.Module):
    """Small MLP wrapper shared by actor/critic policy modules."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, activation: str = "relu") -> None:
        super().__init__()
        self.network = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            output_dim=None,
        )
        self.output_dim = hidden_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
