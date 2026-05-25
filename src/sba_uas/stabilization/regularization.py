"""Parameter-stabilization regularizers for Critic and SAN."""

from __future__ import annotations

from typing import Dict, Mapping, Union

import torch
from torch import nn

from sba_uas.stabilization.reference_network import ReferenceNetwork


ImportanceMap = Mapping[str, torch.Tensor]


def make_importance_like(
    model: nn.Module,
    fill_value: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Create an Omega map with one tensor per trainable parameter."""

    return {
        name: torch.full_like(param.detach(), float(fill_value))
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def parameter_stabilization_loss(
    model: nn.Module,
    reference: Union[nn.Module, ReferenceNetwork],
    importance: ImportanceMap,
    eta: float,
) -> torch.Tensor:
    """Compute ``eta * sum_i Omega_i * (theta_i - theta_ref_i)^2``."""

    if eta < 0.0:
        raise ValueError("eta must be non-negative")

    ref_parameters = _reference_parameter_dict(reference)
    total = None
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name not in ref_parameters:
            raise KeyError("reference is missing parameter '{}'".format(name))
        if name not in importance:
            raise KeyError("importance is missing parameter '{}'".format(name))

        ref_param = ref_parameters[name].detach().to(device=param.device, dtype=param.dtype)
        omega = importance[name].detach().to(device=param.device, dtype=param.dtype)
        if tuple(ref_param.shape) != tuple(param.shape):
            raise ValueError("reference shape mismatch for '{}'".format(name))
        if tuple(omega.shape) != tuple(param.shape):
            raise ValueError("importance shape mismatch for '{}'".format(name))

        value = (omega.clamp_min(0.0) * (param - ref_param).pow(2)).sum()
        total = value if total is None else total + value

    if total is None:
        first_param = next(model.parameters())
        total = first_param.new_tensor(0.0)
    return total * float(eta)


def _reference_parameter_dict(
    reference: Union[nn.Module, ReferenceNetwork],
) -> Dict[str, torch.Tensor]:
    if isinstance(reference, ReferenceNetwork):
        return reference.parameter_dict()
    return {name: param for name, param in reference.named_parameters()}
