"""Checkpoint helpers for Roach-compatible SBA-UAS training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn


def save_roach_policy_checkpoint(
    path: Path,
    policy: nn.Module,
    train_init_kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Save the policy-only checkpoint expected by Roach ``PpoPolicy.load``."""

    if not hasattr(policy, "get_init_kwargs"):
        raise TypeError("policy must expose get_init_kwargs()")

    get_init_kwargs = getattr(policy, "get_init_kwargs")
    # Roach PpoPolicy.load() expects this compact contract. Keep all SBA-UAS
    # sidecar state out of this file so upstream evaluation can still load it.
    payload = {
        "policy_state_dict": policy.state_dict(),
        "policy_init_kwargs": get_init_kwargs(),
        "train_init_kwargs": dict(train_init_kwargs),
    }
    _save_payload(path, payload)
    return payload


def save_sba_uas_extra_state(
    path: Path,
    *,
    critic: Optional[nn.Module] = None,
    san: Optional[nn.Module] = None,
    environment_model: Optional[nn.Module] = None,
    reference_critic: Optional[nn.Module] = None,
    reference_san: Optional[nn.Module] = None,
    standard_buffer: Optional[Any] = None,
    familiar_buffer: Optional[Any] = None,
    critic_importance: Optional[Mapping[str, torch.Tensor]] = None,
    san_importance: Optional[Mapping[str, torch.Tensor]] = None,
    reward_parameter_correlation: Optional[Any] = None,
    san_synaptic_importance: Optional[Any] = None,
    optimizers: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Save SBA-UAS state that must not be mixed into Roach policy files."""

    # The section names mirror the restore paths in trainer.load_extra_state().
    # That makes partial checkpoints possible while keeping the payload readable.
    payload: Dict[str, Any] = {
        "metadata": dict(metadata or {}),
        "modules": {},
        "buffers": {},
        "importance": {},
        "trackers": {},
        "optimizers": {},
    }

    _add_module_state(payload["modules"], "critic", critic)
    _add_module_state(payload["modules"], "san", san)
    _add_module_state(payload["modules"], "environment_model", environment_model)
    _add_module_state(payload["modules"], "reference_critic", reference_critic)
    _add_module_state(payload["modules"], "reference_san", reference_san)
    _add_stateful(payload["buffers"], "standard", standard_buffer)
    _add_stateful(payload["buffers"], "familiar", familiar_buffer)

    if critic_importance is not None:
        payload["importance"]["critic"] = _clone_tensor_mapping(critic_importance)
    if san_importance is not None:
        payload["importance"]["san"] = _clone_tensor_mapping(san_importance)
    _add_stateful(
        payload["trackers"],
        "reward_parameter_correlation",
        reward_parameter_correlation,
    )
    _add_stateful(payload["trackers"], "san_synaptic_importance", san_synaptic_importance)
    if optimizers is not None:
        # Optimizer state_dicts already own their tensor structure, so preserve
        # them verbatim instead of trying to clone or normalize nested values.
        payload["optimizers"] = {
            name: optimizer_state
            for name, optimizer_state in optimizers.items()
        }

    _save_payload(path, payload)
    return payload


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint payload from disk."""

    return torch.load(str(path), map_location=map_location)


def _add_module_state(
    target: Dict[str, Any],
    name: str,
    module: Optional[nn.Module],
) -> None:
    if module is not None:
        target[name] = module.state_dict()


def _add_stateful(target: Dict[str, Any], name: str, value: Optional[Any]) -> None:
    if value is None:
        return
    if not hasattr(value, "state_dict"):
        raise TypeError("{} must expose state_dict()".format(name))
    target[name] = value.state_dict()


def _clone_tensor_mapping(values: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in values.items()}


def _save_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Materialize the mapping so later caller mutations cannot alter the object
    # that torch is serializing.
    torch.save(dict(payload), str(path))
