"""Gated Double-Q Critic for SBA-UAS."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class GatedQNetwork(nn.Module):
    """One Q network whose hidden activations are modulated by SAN gates."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        measurement_dim: int = 0,
        hidden_units: int = 2048,
        n_layers: int = 6,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if measurement_dim < 0:
            raise ValueError("measurement_dim cannot be negative")
        if hidden_units <= 0:
            raise ValueError("hidden_units must be positive")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.measurement_dim = measurement_dim
        self.hidden_units = hidden_units
        self.n_layers = n_layers

        input_dim = state_dim + measurement_dim + action_dim
        dims = [input_dim] + [hidden_units] * n_layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(dims[index], dims[index + 1]) for index in range(n_layers)]
        )
        self.output_layer = nn.Linear(hidden_units, 1)

    def forward(
        self,
        state: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._compose_input(state=state, measurement=measurement, action=action)
        self._check_gates(gates, batch_size=hidden.shape[0])

        for layer_index, layer in enumerate(self.hidden_layers):
            hidden = F.relu(layer(hidden))
            # SAN gates are shaped [batch, layer, unit], so each Critic layer can
            # reuse a different sparse sub-network for the same transition.
            hidden = hidden * gates[:, layer_index, :]
        return self.output_layer(hidden).squeeze(-1)

    def _compose_input(
        self,
        state: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        state = state.reshape(state.shape[0], -1)
        action = action.reshape(action.shape[0], -1)
        if state.shape[1] != self.state_dim:
            raise ValueError(
                "state dim mismatch: expected {}, got {}".format(
                    self.state_dim, state.shape[1]
                )
            )
        if action.shape[1] != self.action_dim:
            raise ValueError(
                "action dim mismatch: expected {}, got {}".format(
                    self.action_dim, action.shape[1]
                )
            )
        if action.shape[0] != state.shape[0]:
            raise ValueError("state and action batch sizes differ")

        parts = [state]
        if self.measurement_dim:
            if measurement is None:
                raise ValueError("measurement is required when measurement_dim > 0")
            measurement = measurement.reshape(measurement.shape[0], -1)
            if measurement.shape[1] != self.measurement_dim:
                raise ValueError(
                    "measurement dim mismatch: expected {}, got {}".format(
                        self.measurement_dim, measurement.shape[1]
                    )
                )
            if measurement.shape[0] != state.shape[0]:
                raise ValueError("state and measurement batch sizes differ")
            parts.append(measurement)
        elif measurement is not None and measurement.numel() > 0:
            raise ValueError("measurement was provided but measurement_dim is 0")

        parts.append(action)
        return torch.cat(parts, dim=1)

    def _check_gates(self, gates: torch.Tensor, batch_size: int) -> None:
        expected_shape = (batch_size, self.n_layers, self.hidden_units)
        if tuple(gates.shape) != expected_shape:
            raise ValueError(
                "gates shape mismatch: expected {}, got {}".format(
                    expected_shape, tuple(gates.shape)
                )
            )


class GatedDoubleCritic(nn.Module):
    """Twin Gated Q networks for clipped double-Q style losses."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        measurement_dim: int = 0,
        hidden_units: int = 2048,
        n_layers: int = 6,
    ) -> None:
        super().__init__()
        self.q1 = GatedQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            measurement_dim=measurement_dim,
            hidden_units=hidden_units,
            n_layers=n_layers,
        )
        self.q2 = GatedQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            measurement_dim=measurement_dim,
            hidden_units=hidden_units,
            n_layers=n_layers,
        )

    def forward(
        self,
        state: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
        gates: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.q1(state=state, measurement=measurement, action=action, gates=gates),
            self.q2(state=state, measurement=measurement, action=action, gates=gates),
        )
