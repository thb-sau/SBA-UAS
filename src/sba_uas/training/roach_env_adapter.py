"""Adapters between Roach rollouts and SBA-UAS training batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

from sba_uas.stabilization.replay_buffer import Transition


@dataclass(frozen=True)
class RoachObservationBatch:
    """Tensor view of Roach wrapper observations."""

    birdview: torch.Tensor
    measurement: torch.Tensor


@dataclass(frozen=True)
class TransitionTensorBatch:
    """Stacked tensor batch sampled from SBA-UAS replay buffers."""

    state: torch.Tensor
    measurement: Optional[torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    next_measurement: Optional[torch.Tensor]
    done: torch.Tensor
    u_vas: torch.Tensor

    def to_policy_obs(self) -> Dict[str, np.ndarray]:
        """Return a Roach-compatible obs dict for ``PpoPolicy.forward``."""

        obs = {
            "birdview": self.state.detach().cpu().numpy(),
        }
        if self.measurement is not None:
            obs["state"] = self.measurement.detach().cpu().numpy()
        return obs

    def next_to_policy_obs(self) -> Dict[str, np.ndarray]:
        """Return next-state obs dict for ``PpoPolicy.forward``."""

        obs = {
            "birdview": self.next_state.detach().cpu().numpy(),
        }
        if self.next_measurement is not None:
            obs["state"] = self.next_measurement.detach().cpu().numpy()
        return obs


class RoachObservationAdapter:
    """Convert Roach ``RlBirdviewWrapper`` observations into tensors."""

    def __init__(
        self,
        device: str = "cpu",
        birdview_key: str = "birdview",
        measurement_key: str = "state",
    ) -> None:
        self.device = torch.device(device)
        self.birdview_key = birdview_key
        self.measurement_key = measurement_key

    def to_tensor_batch(self, obs: Mapping[str, Any]) -> RoachObservationBatch:
        """Convert a Roach obs dict to batched tensors."""

        if self.birdview_key not in obs:
            raise KeyError("obs is missing '{}'".format(self.birdview_key))
        if self.measurement_key not in obs:
            raise KeyError("obs is missing '{}'".format(self.measurement_key))

        birdview = _as_tensor(obs[self.birdview_key], self.device)
        measurement = _as_tensor(obs[self.measurement_key], self.device).float()
        if birdview.dim() == 3:
            birdview = birdview.unsqueeze(0)
        if measurement.dim() == 1:
            measurement = measurement.unsqueeze(0)
        if birdview.dim() != 4:
            raise ValueError("birdview must have shape [batch, channels, height, width]")
        if measurement.dim() != 2:
            raise ValueError("measurement must have shape [batch, dim]")
        if birdview.shape[0] != measurement.shape[0]:
            raise ValueError("birdview and measurement batch sizes differ")
        return RoachObservationBatch(birdview=birdview, measurement=measurement)


class RoachTransitionAdapter:
    """Annotate Roach environment steps with shifted VAS and build transitions."""

    def __init__(
        self,
        observation_adapter: Optional[RoachObservationAdapter] = None,
    ) -> None:
        self.observation_adapter = observation_adapter or RoachObservationAdapter()

    def annotate_step(
        self,
        obs: Mapping[str, Any],
        actions: Any,
        rewards: Any,
        next_obs: Mapping[str, Any],
        dones: Any,
        infos: Optional[Sequence[Mapping[str, Any]]],
        annotator: Any,
    ) -> List[Transition]:
        """Convert one vectorized Roach step into annotated transitions."""

        current = self.observation_adapter.to_tensor_batch(obs)
        next_batch = self.observation_adapter.to_tensor_batch(next_obs)
        action_tensor = _as_tensor(actions, self.observation_adapter.device).float()
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
        # Rewards and dones may arrive as scalars in single-env smoke tests or
        # as vectors in Roach's vectorized rollout path.
        reward_values = _as_numpy_1d(rewards)
        done_values = _as_numpy_1d(dones).astype(np.bool_)

        batch_size = current.birdview.shape[0]
        if action_tensor.shape[0] != batch_size:
            raise ValueError("action and observation batch sizes differ")
        if next_batch.birdview.shape[0] != batch_size:
            raise ValueError("next observation batch size differs")
        if reward_values.shape[0] != batch_size or done_values.shape[0] != batch_size:
            raise ValueError("reward/done batch sizes differ from observations")

        info_values = list(infos or [{} for _ in range(batch_size)])
        if len(info_values) != batch_size:
            raise ValueError("infos length differs from observations")

        transitions = []
        for index in range(batch_size):
            transitions.append(
                annotator.annotate(
                    state=current.birdview[index : index + 1],
                    measurement=current.measurement[index : index + 1],
                    action=action_tensor[index : index + 1],
                    reward=float(reward_values[index]),
                    next_state=next_batch.birdview[index : index + 1],
                    next_measurement=next_batch.measurement[index : index + 1],
                    done=bool(done_values[index]),
                    metadata=_metadata_from_info(info_values[index]),
                )
            )
        return transitions


def transitions_to_tensor_batch(
    transitions: Iterable[Transition],
    device: str = "cpu",
) -> TransitionTensorBatch:
    """Stack replay-buffer transitions into tensors for model updates."""

    items = list(transitions)
    if not items:
        raise ValueError("transitions cannot be empty")
    target_device = torch.device(device)
    state = _stack_values([item.state for item in items], target_device)
    action = _stack_values([item.action for item in items], target_device).float()
    next_state = _stack_values([item.next_state for item in items], target_device)
    reward = torch.as_tensor(
        [float(item.reward) for item in items],
        dtype=torch.float32,
    ).to(device=target_device)
    done = torch.as_tensor(
        [float(item.done) for item in items],
        dtype=torch.float32,
    ).to(device=target_device)
    u_vas = torch.as_tensor(
        [float(item.u_vas) for item in items],
        dtype=torch.float32,
    ).to(device=target_device)

    measurements = [item.measurement for item in items]
    next_measurements = [item.next_measurement for item in items]
    measurement = None
    next_measurement = None
    # Mixed presence would silently produce malformed Roach policy inputs, so
    # fail before stacking instead of filling missing measurements.
    if any(value is not None for value in measurements):
        if any(value is None for value in measurements):
            raise ValueError("measurements must be all present or all None")
        measurement = _stack_values(measurements, target_device).float()
    if any(value is not None for value in next_measurements):
        if any(value is None for value in next_measurements):
            raise ValueError("next_measurements must be all present or all None")
        next_measurement = _stack_values(next_measurements, target_device).float()

    return TransitionTensorBatch(
        state=state,
        measurement=measurement,
        action=action,
        reward=reward,
        next_state=next_state,
        next_measurement=next_measurement,
        done=done,
        u_vas=u_vas,
    )


def _as_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return torch.as_tensor(value).to(device=device)


def _as_numpy_1d(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(-1)


def _stack_values(values: Sequence[Any], device: torch.device) -> torch.Tensor:
    tensors = []
    for value in values:
        tensor = _as_tensor(value, device)
        tensors.append(tensor)
    return torch.stack(tensors, dim=0)


def _metadata_from_info(info: Mapping[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key in ("town", "map", "route_id", "task_idx", "scenario"):
        if key in info:
            metadata[key] = info[key]
    if "episode_stat" in info:
        metadata["episode_stat"] = info["episode_stat"]
    return metadata
