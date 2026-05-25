"""SI-style parameter-importance tracking for SAN stabilization."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import nn


class SynapticIntelligenceImportance:
    """Track importance from parameter movement aligned with loss gradients.

    This matches the SAN-side importance rule in the paper:
    ``omega += delta_theta * (-grad_L)``, normalized by the absolute path
    contribution and clipped at zero when converted to ``Omega``.
    """

    def __init__(
        self,
        previous_parameters: Mapping[str, torch.Tensor],
        omega: Mapping[str, torch.Tensor],
        omega_bar: Mapping[str, torch.Tensor],
        damping: float = 1.0e-8,
    ) -> None:
        if damping <= 0.0:
            raise ValueError("damping must be positive")
        self.previous_parameters = _clone_map(previous_parameters)
        self.omega = _clone_map(omega)
        self.omega_bar = _clone_map(omega_bar)
        self.damping = float(damping)
        self._validate_state_keys()

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        damping: float = 1.0e-8,
    ) -> "SynapticIntelligenceImportance":
        parameters = _trainable_parameter_map(model)
        zeros = {name: torch.zeros_like(param) for name, param in parameters.items()}
        return cls(
            previous_parameters=parameters,
            omega=zeros,
            omega_bar=zeros,
            damping=damping,
        )

    def update(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Observe one optimizer step and return the current importance map.

        Call this after ``loss.backward()`` and after the optimizer step, while
        gradients from the loss are still attached to the model parameters.
        """

        current_parameters = _trainable_parameter_map(model)
        self._validate_model_parameters(current_parameters)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            current = current_parameters[name]
            previous = self.previous_parameters[name].to(
                device=current.device,
                dtype=current.dtype,
            )
            delta_param = current - previous
            if param.grad is None:
                gradient = torch.zeros_like(current)
            else:
                gradient = param.grad.detach().to(
                    device=current.device,
                    dtype=current.dtype,
                )
            path_contribution = delta_param * (-gradient)
            self.omega[name] = self.omega[name].to(
                device=current.device,
                dtype=current.dtype,
            ) + path_contribution
            self.omega_bar[name] = self.omega_bar[name].to(
                device=current.device,
                dtype=current.dtype,
            ) + delta_param.abs() * gradient.abs()
            self.previous_parameters[name] = current.detach().clone()

        return self.importance()

    def importance(self) -> Dict[str, torch.Tensor]:
        """Return non-negative ``Omega`` tensors keyed by parameter name."""

        return {
            name: torch.clamp(
                self.omega[name] / (self.omega_bar[name] + self.damping),
                min=0.0,
            )
            for name in self.omega
        }

    def state_dict(self) -> Dict[str, object]:
        """Return checkpoint-friendly tracker state."""

        return {
            "damping": self.damping,
            "previous_parameters": _clone_map(self.previous_parameters),
            "omega": _clone_map(self.omega),
            "omega_bar": _clone_map(self.omega_bar),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore tracker state produced by :meth:`state_dict`."""

        self.damping = float(state_dict["damping"])
        if self.damping <= 0.0:
            raise ValueError("damping must be positive")
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
