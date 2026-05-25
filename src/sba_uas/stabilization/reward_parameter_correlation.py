"""Reward-Parameter Correlation importance tracking."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import nn


class RewardParameterCorrelation:
    """Track Critic parameter importance from reward-correlated updates.

    For each trainable parameter ``phi_i`` this class accumulates
    ``omega_i += delta_phi_i * delta_reward`` and
    ``omega_bar_i += |delta_phi_i| * |delta_reward|``. The normalized
    importance map is ``Omega_i = omega_i / (omega_bar_i + damping)``.
    """

    def __init__(
        self,
        previous_parameters: Mapping[str, torch.Tensor],
        omega: Mapping[str, torch.Tensor],
        omega_bar: Mapping[str, torch.Tensor],
        damping: float = 1.0e-8,
        previous_reward: Optional[float] = None,
    ) -> None:
        if damping <= 0.0:
            raise ValueError("damping must be positive")
        self.previous_parameters = _clone_map(previous_parameters)
        self.omega = _clone_map(omega)
        self.omega_bar = _clone_map(omega_bar)
        self.damping = float(damping)
        self.previous_reward = previous_reward
        self._validate_state_keys()

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        damping: float = 1.0e-8,
    ) -> "RewardParameterCorrelation":
        parameters = _trainable_parameter_map(model)
        zeros = {name: torch.zeros_like(param) for name, param in parameters.items()}
        return cls(
            previous_parameters=parameters,
            omega=zeros,
            omega_bar=zeros,
            damping=damping,
            previous_reward=None,
        )

    def update(
        self,
        model: nn.Module,
        batch_mean_reward,
    ) -> Dict[str, torch.Tensor]:
        """Observe one training step and return the current importance map."""

        reward = _as_reward_float(batch_mean_reward)
        current_parameters = _trainable_parameter_map(model)
        self._validate_model_parameters(current_parameters)

        if self.previous_reward is None:
            self.previous_parameters = _clone_map(current_parameters)
            self.previous_reward = reward
            return self.importance()

        reward_delta = reward - self.previous_reward
        abs_reward_delta = abs(reward_delta)
        for name, current in current_parameters.items():
            previous = self.previous_parameters[name].to(
                device=current.device,
                dtype=current.dtype,
            )
            delta_param = current - previous
            self.omega[name] = self.omega[name].to(
                device=current.device,
                dtype=current.dtype,
            ) + delta_param * reward_delta
            self.omega_bar[name] = self.omega_bar[name].to(
                device=current.device,
                dtype=current.dtype,
            ) + delta_param.abs() * abs_reward_delta
            self.previous_parameters[name] = current.detach().clone()

        self.previous_reward = reward
        return self.importance()

    def importance(self) -> Dict[str, torch.Tensor]:
        """Return ``Omega`` tensors keyed by model parameter name."""

        return {
            name: self.omega[name] / (self.omega_bar[name] + self.damping)
            for name in self.omega
        }

    def state_dict(self) -> Dict[str, object]:
        """Return checkpoint-friendly tracker state."""

        return {
            "damping": self.damping,
            "previous_reward": self.previous_reward,
            "previous_parameters": _clone_map(self.previous_parameters),
            "omega": _clone_map(self.omega),
            "omega_bar": _clone_map(self.omega_bar),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore tracker state produced by :meth:`state_dict`."""

        self.damping = float(state_dict["damping"])
        if self.damping <= 0.0:
            raise ValueError("damping must be positive")
        previous_reward = state_dict.get("previous_reward")
        self.previous_reward = (
            None if previous_reward is None else float(previous_reward)
        )
        self.previous_parameters = _clone_map(state_dict["previous_parameters"])
        self.omega = _clone_map(state_dict["omega"])
        self.omega_bar = _clone_map(state_dict["omega_bar"])
        self._validate_state_keys()

    def _validate_model_parameters(
        self, parameters: Mapping[str, torch.Tensor]
    ) -> None:
        expected = set(self.previous_parameters)
        actual = set(parameters)
        if expected != actual:
            raise KeyError(
                "model parameter names changed: expected {}, got {}".format(
                    sorted(expected),
                    sorted(actual),
                )
            )
        for name, current in parameters.items():
            if tuple(current.shape) != tuple(self.previous_parameters[name].shape):
                raise ValueError("parameter shape changed for '{}'".format(name))

    def _validate_state_keys(self) -> None:
        keys = set(self.previous_parameters)
        if keys != set(self.omega) or keys != set(self.omega_bar):
            raise KeyError("previous_parameters, omega, and omega_bar keys must match")


def _trainable_parameter_map(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _clone_map(values: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in values.items()}


def _as_reward_float(value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("batch_mean_reward tensor must contain one value")
        return float(value.detach().cpu().item())
    return float(value)
