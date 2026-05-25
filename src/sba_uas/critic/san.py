"""Similarity-Based Activation Network for Critic-side gating."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


class SimilarityBasedActivationNetwork(nn.Module):
    """Produce sparse sigmoid gates for gated Critic hidden layers."""

    def __init__(
        self,
        state_dim: int,
        measurement_dim: int = 0,
        n_layers: int = 6,
        units_per_layer: int = 2048,
        feature_dim: int = 256,
        shared_hidden_units: int = 1024,
        target_active_ratio: float = 0.2,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if measurement_dim < 0:
            raise ValueError("measurement_dim cannot be negative")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if units_per_layer <= 0:
            raise ValueError("units_per_layer must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if shared_hidden_units <= 0:
            raise ValueError("shared_hidden_units must be positive")
        if not 0.0 < target_active_ratio < 1.0:
            raise ValueError("target_active_ratio must be in (0, 1)")

        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.n_layers = n_layers
        self.units_per_layer = units_per_layer
        self.target_active_ratio = float(target_active_ratio)

        input_dim = state_dim + measurement_dim
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, shared_hidden_units),
            nn.ReLU(),
            nn.Linear(shared_hidden_units, feature_dim),
            nn.ReLU(),
        )
        self.gate_hidden = nn.Sequential(
            nn.Linear(feature_dim, shared_hidden_units),
            nn.ReLU(),
        )
        self.gate_logits = nn.Linear(shared_hidden_units, n_layers * units_per_layer)
        self._initialize_gate_head()

    def forward(
        self, state: torch.Tensor, measurement: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self._compose_input(state=state, measurement=measurement)
        features = self.feature_encoder(inputs)
        logits = self.gate_logits(self.gate_hidden(features))
        gates = torch.sigmoid(logits).view(
            state.shape[0],
            self.n_layers,
            self.units_per_layer,
        )
        return gates, features

    def _initialize_gate_head(self) -> None:
        nn.init.normal_(self.gate_logits.weight, mean=0.0, std=1.0e-3)
        bias = math.log(self.target_active_ratio / (1.0 - self.target_active_ratio))
        nn.init.constant_(self.gate_logits.bias, bias)

    def _compose_input(
        self, state: torch.Tensor, measurement: Optional[torch.Tensor]
    ) -> torch.Tensor:
        state = state.reshape(state.shape[0], -1)
        if state.shape[1] != self.state_dim:
            raise ValueError(
                "state dim mismatch: expected {}, got {}".format(
                    self.state_dim, state.shape[1]
                )
            )

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
            return torch.cat([state, measurement], dim=1)

        if measurement is not None and measurement.numel() > 0:
            raise ValueError("measurement was provided but measurement_dim is 0")
        return state
