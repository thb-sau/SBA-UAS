"""Roach-compatible SBA-UAS sidecar trainer.

This trainer deliberately keeps the Roach Actor/policy object intact. It uses
the policy only to collect actions and to produce bootstrap actions; all
SBA-UAS state is trained and checkpointed separately as Critic/training-side
extra state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn

from sba_uas.critic.gated_critic import GatedDoubleCritic
from sba_uas.critic.losses import (
    clipped_double_q_target,
    double_q_bellman_loss,
    san_loss,
)
from sba_uas.critic.san import SimilarityBasedActivationNetwork
from sba_uas.stabilization.environment_model import BayesianBEVEnvironmentModel
from sba_uas.stabilization.reference_network import ReferenceNetwork
from sba_uas.stabilization.regularization import parameter_stabilization_loss
from sba_uas.stabilization.replay_buffer import (
    FamiliarExperienceBuffer,
    StandardReplayBuffer,
    Transition,
)
from sba_uas.stabilization.reward_parameter_correlation import (
    RewardParameterCorrelation,
)
from sba_uas.stabilization.synaptic_importance import (
    SynapticIntelligenceImportance,
)
from sba_uas.stabilization.vas import ShiftedVASScorer, VASTransitionAnnotator
from sba_uas.training.checkpointing import (
    load_checkpoint,
    save_roach_policy_checkpoint,
    save_sba_uas_extra_state,
)
from sba_uas.training.roach_env_adapter import (
    RoachObservationAdapter,
    RoachTransitionAdapter,
    TransitionTensorBatch,
    transitions_to_tensor_batch,
)


@dataclass
class SBAUASTrainerConfig:
    """Configuration for the Roach-compatible sidecar trainer."""

    device: str = "cpu"
    standard_buffer_capacity: int = 4096
    familiar_buffer_capacity: int = 4096
    batch_size: int = 32
    environment_batch_size: int = 32
    reference_batch_size: int = 32
    warmup_transitions: int = 256
    updates_per_env_step: int = 1
    discount: float = 0.99
    deterministic_rollout: bool = False
    clip_policy_actions: bool = False
    target_active_ratio: float = 0.2
    san_similarity_weight: float = 1.0
    san_activation_weight: float = 1.0
    critic_stabilization_eta: float = 1.0e-3
    san_stabilization_eta: float = 1.0e-3
    environment_kl_weight: float = 1.0e-4
    environment_prediction_samples: int = 1
    shifted_vas_samples: int = 64
    shifted_vas_sigma: float = 1.0
    learning_rate: float = 1.0e-4
    reference_learning_rate: float = 1.0e-4
    environment_learning_rate: float = 1.0e-4
    max_grad_norm: Optional[float] = 10.0
    checkpoint_interval_transitions: Optional[int] = None
    policy_checkpoint_path: Optional[Path] = None
    extra_checkpoint_path: Optional[Path] = None
    train_init_kwargs: Mapping[str, Any] = field(default_factory=dict)
    rng_seed: int = 2021


class RoachCompatibleSBAUASTrainer:
    """Train SBA-UAS extra state around an unmodified Roach policy."""

    def __init__(
        self,
        policy: nn.Module,
        env: Any,
        config: Optional[SBAUASTrainerConfig] = None,
        *,
        environment_model: Optional[BayesianBEVEnvironmentModel] = None,
        san: Optional[SimilarityBasedActivationNetwork] = None,
        critic: Optional[GatedDoubleCritic] = None,
        reference_san: Optional[ReferenceNetwork] = None,
        reference_critic: Optional[ReferenceNetwork] = None,
        standard_buffer: Optional[StandardReplayBuffer] = None,
        familiar_buffer: Optional[FamiliarExperienceBuffer] = None,
        observation_adapter: Optional[RoachObservationAdapter] = None,
        hidden_units: int = 2048,
        n_layers: int = 6,
        san_feature_dim: int = 256,
        san_shared_hidden_units: int = 1024,
        environment_base_channels: int = 32,
        environment_transition_hidden_dims: Iterable[int] = (512, 512),
        latent_dim: int = 256,
    ) -> None:
        self.policy = policy
        self.env = env
        self.config = config or SBAUASTrainerConfig()
        self.device = torch.device(self.config.device)
        self.rng = random.Random(self.config.rng_seed)
        self.num_timesteps = 0
        self._last_obs = None
        self.last_train_metrics: Dict[str, float] = {}

        # Roach exposes BEV, measurement, and action dimensions through Gym
        # spaces. Infer them once so the sidecar stays independent of Roach code.
        bev_shape, measurement_dim, action_dim = _infer_roach_shapes(env)
        self.environment_model = environment_model or BayesianBEVEnvironmentModel(
            bev_shape=bev_shape,
            action_dim=action_dim,
            measurement_dim=measurement_dim,
            latent_dim=latent_dim,
            base_channels=environment_base_channels,
            transition_hidden_dims=environment_transition_hidden_dims,
        )
        self.san = san or SimilarityBasedActivationNetwork(
            state_dim=latent_dim,
            measurement_dim=measurement_dim,
            n_layers=n_layers,
            units_per_layer=hidden_units,
            feature_dim=san_feature_dim,
            shared_hidden_units=san_shared_hidden_units,
            target_active_ratio=self.config.target_active_ratio,
        )
        self.critic = critic or GatedDoubleCritic(
            state_dim=latent_dim,
            action_dim=action_dim,
            measurement_dim=measurement_dim,
            hidden_units=hidden_units,
            n_layers=n_layers,
        )

        self.environment_model.to(self.device)
        self.san.to(self.device)
        self.critic.to(self.device)

        self.reference_san = reference_san or ReferenceNetwork.from_model(self.san)
        self.reference_critic = reference_critic or ReferenceNetwork.from_model(self.critic)
        self.reference_san.to(self.device)
        self.reference_critic.to(self.device)
        # ReferenceNetwork freezes snapshots by default; SBA-UAS updates these
        # copies on D-only familiar data, so their parameters must be trainable.
        self.reference_san.unfreeze()
        self.reference_critic.unfreeze()

        self.familiar_buffer = familiar_buffer or FamiliarExperienceBuffer(
            capacity=self.config.familiar_buffer_capacity,
            rng=self.rng,
        )
        self.standard_buffer = standard_buffer or StandardReplayBuffer(
            capacity=self.config.standard_buffer_capacity,
            familiar_buffer=self.familiar_buffer,
            rng=self.rng,
        )

        self.environment_optimizer = torch.optim.Adam(
            self.environment_model.parameters(),
            lr=self.config.environment_learning_rate,
        )
        self.san_optimizer = torch.optim.Adam(
            self.san.parameters(),
            lr=self.config.learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.learning_rate,
        )
        self.reference_san_optimizer = torch.optim.Adam(
            self.reference_san.model.parameters(),
            lr=self.config.reference_learning_rate,
        )
        self.reference_critic_optimizer = torch.optim.Adam(
            self.reference_critic.model.parameters(),
            lr=self.config.reference_learning_rate,
        )

        self.critic_importance = RewardParameterCorrelation.from_model(self.critic)
        self.san_importance = SynapticIntelligenceImportance.from_model(self.san)
        self.vas_scorer = ShiftedVASScorer(
            monte_carlo_samples=self.config.shifted_vas_samples,
            sigma=self.config.shifted_vas_sigma,
        )
        self.vas_annotator = VASTransitionAnnotator(
            model=self.environment_model,
            scorer=self.vas_scorer,
        )
        self.observation_adapter = observation_adapter or RoachObservationAdapter(
            device=self.config.device,
        )
        self.transition_adapter = RoachTransitionAdapter(self.observation_adapter)

    def learn(self, total_timesteps: int, seed: Optional[int] = None) -> "RoachCompatibleSBAUASTrainer":
        """Collect Roach rollouts and update SBA-UAS extra state."""

        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if seed is not None:
            self.seed(seed)
        if self._last_obs is None:
            self._last_obs = self.env.reset()

        while self.num_timesteps < total_timesteps:
            collected = self.collect_roach_rollout(n_steps=1)
            # The environment model first needs enough annotated transitions for
            # shifted-VAS scores and buffer migration to be meaningful.
            if len(self.standard_buffer) >= self.config.warmup_transitions:
                for _ in range(self.config.updates_per_env_step):
                    self.last_train_metrics = self.train_step()
            self._maybe_save_checkpoint()
            if collected == 0:
                break
        return self

    def seed(self, seed: int) -> None:
        """Seed local RNGs and best-effort Roach env spaces."""

        self.rng.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
        for space_name in ("action_space", "observation_space"):
            space = getattr(self.env, space_name, None)
            if hasattr(space, "seed"):
                space.seed(seed)

    def collect_roach_rollout(self, n_steps: int = 1) -> int:
        """Use the unchanged Roach policy to collect and annotate transitions."""

        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self._last_obs is None:
            self._last_obs = self.env.reset()

        collected = 0
        self.policy.eval()
        for _ in range(n_steps):
            # Actions always come from the upstream policy; SBA-UAS only observes
            # and annotates the resulting transition.
            actions = self._policy_actions(
                self._last_obs,
                deterministic=self.config.deterministic_rollout,
                clip_action=self.config.clip_policy_actions,
            )
            next_obs, rewards, dones, infos = self.env.step(actions)
            transitions = self.transition_adapter.annotate_step(
                obs=self._last_obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
                infos=infos,
                annotator=self.vas_annotator,
            )
            for transition in transitions:
                self.standard_buffer.add(transition)
            collected += len(transitions)
            self.num_timesteps += len(transitions)
            self._last_obs = next_obs
        return collected

    def train_step(self) -> Dict[str, float]:
        """Run one SBA-UAS update cycle."""

        metrics: Dict[str, float] = {}
        metrics.update(self.update_environment_model())
        metrics.update(self.update_reference_networks())
        metrics.update(self.update_main_networks())
        return metrics

    def update_environment_model(self) -> Dict[str, float]:
        """Update the BEV autoencoder + BNN from ``B union D``."""

        transitions = self._sample_union(self.config.environment_batch_size)
        if not transitions:
            return {}
        batch = transitions_to_tensor_batch(transitions, device=self.config.device)
        self.environment_model.train()
        self.environment_optimizer.zero_grad()
        loss = self.environment_model.training_loss(
            state=batch.state,
            measurement=batch.measurement,
            action=batch.action,
            next_state=batch.next_state,
            kl_weight=self.config.environment_kl_weight,
            prediction_samples=self.config.environment_prediction_samples,
        )
        loss.backward()
        self._clip_gradients(self.environment_model)
        self.environment_optimizer.step()
        return {"env_model/loss": float(loss.detach().cpu().item())}

    def update_reference_networks(self) -> Dict[str, float]:
        """Train reference Critic/SAN on familiar buffer ``D`` only."""

        # Reference networks represent stable old-domain behavior, so they only
        # learn from transitions that the uncertainty filter marked as familiar.
        transitions = self._sample_from_buffer(
            list(self.familiar_buffer),
            self.config.reference_batch_size,
        )
        if not transitions:
            return {}
        batch = transitions_to_tensor_batch(transitions, device=self.config.device)
        latent, next_latent = self._encoded_pair(batch)

        self.reference_san_optimizer.zero_grad()
        reference_gates, reference_features = self.reference_san.model(
            latent,
            batch.measurement,
        )
        reference_san_loss = san_loss(
            gates=reference_gates,
            state_features=reference_features,
            target_active_ratio=self.config.target_active_ratio,
            similarity_weight=self.config.san_similarity_weight,
            activation_weight=self.config.san_activation_weight,
        )
        reference_san_loss.backward()
        self._clip_gradients(self.reference_san.model)
        self.reference_san_optimizer.step()

        # Bootstrap targets use the unchanged Roach policy for the next action,
        # matching the sidecar constraint that Actor structure is untouched.
        next_actions = self._policy_actions_for_batch(batch, next_state=True)
        with torch.no_grad():
            next_gates, _ = self.reference_san.model(next_latent, batch.next_measurement)
            next_q1, next_q2 = self.reference_critic.model(
                next_latent,
                batch.next_measurement,
                next_actions,
                next_gates,
            )
            target_q = clipped_double_q_target(
                next_q1=next_q1,
                next_q2=next_q2,
                reward=batch.reward,
                done=batch.done,
                discount=self.config.discount,
            )

        self.reference_critic_optimizer.zero_grad()
        with torch.no_grad():
            current_gates, _ = self.reference_san.model(latent, batch.measurement)
        q1, q2 = self.reference_critic.model(
            latent,
            batch.measurement,
            batch.action,
            current_gates,
        )
        reference_critic_loss = double_q_bellman_loss(q1, q2, target_q)
        reference_critic_loss.backward()
        self._clip_gradients(self.reference_critic.model)
        self.reference_critic_optimizer.step()

        return {
            "reference/san_loss": float(reference_san_loss.detach().cpu().item()),
            "reference/critic_loss": float(reference_critic_loss.detach().cpu().item()),
        }

    def update_main_networks(self) -> Dict[str, float]:
        """Update SAN and Critic from standard buffer ``B``."""

        # Main SAN/Critic updates follow recent experience from B, while
        # parameter stabilization pulls protected parameters toward D references.
        transitions = self._sample_from_buffer(
            list(self.standard_buffer),
            self.config.batch_size,
        )
        if not transitions:
            return {}
        batch = transitions_to_tensor_batch(transitions, device=self.config.device)
        latent, next_latent = self._encoded_pair(batch)

        self.san_optimizer.zero_grad()
        gates, features = self.san(latent, batch.measurement)
        base_san_loss = san_loss(
            gates=gates,
            state_features=features,
            target_active_ratio=self.config.target_active_ratio,
            similarity_weight=self.config.san_similarity_weight,
            activation_weight=self.config.san_activation_weight,
        )
        san_stabilization = parameter_stabilization_loss(
            model=self.san,
            reference=self.reference_san,
            importance=self.san_importance.importance(),
            eta=self.config.san_stabilization_eta,
        )
        total_san_loss = base_san_loss + san_stabilization
        total_san_loss.backward()
        self._clip_gradients(self.san)
        self.san_optimizer.step()
        san_importance = self.san_importance.update(self.san)

        next_actions = self._policy_actions_for_batch(batch, next_state=True)
        with torch.no_grad():
            next_gates, _ = self.san(next_latent, batch.next_measurement)
            next_q1, next_q2 = self.critic(
                next_latent,
                batch.next_measurement,
                next_actions,
                next_gates,
            )
            target_q = clipped_double_q_target(
                next_q1=next_q1,
                next_q2=next_q2,
                reward=batch.reward,
                done=batch.done,
                discount=self.config.discount,
            )
            current_gates, _ = self.san(latent, batch.measurement)

        self.critic_optimizer.zero_grad()
        q1, q2 = self.critic(
            latent,
            batch.measurement,
            batch.action,
            current_gates,
        )
        critic_bellman_loss = double_q_bellman_loss(q1, q2, target_q)
        critic_stabilization = parameter_stabilization_loss(
            model=self.critic,
            reference=self.reference_critic,
            importance=self.critic_importance.importance(),
            eta=self.config.critic_stabilization_eta,
        )
        total_critic_loss = critic_bellman_loss + critic_stabilization
        total_critic_loss.backward()
        self._clip_gradients(self.critic)
        self.critic_optimizer.step()
        critic_importance = self.critic_importance.update(
            self.critic,
            batch_mean_reward=batch.reward.mean(),
        )

        return {
            "san/loss": float(total_san_loss.detach().cpu().item()),
            "san/base_loss": float(base_san_loss.detach().cpu().item()),
            "san/stabilization_loss": float(san_stabilization.detach().cpu().item()),
            "san/activation_ratio": float(gates.detach().mean().cpu().item()),
            "san/importance_mean": _importance_mean(san_importance),
            "critic/loss": float(total_critic_loss.detach().cpu().item()),
            "critic/bellman_loss": float(critic_bellman_loss.detach().cpu().item()),
            "critic/stabilization_loss": float(
                critic_stabilization.detach().cpu().item()
            ),
            "critic/importance_mean": _importance_mean(critic_importance),
            "replay/u_vas_mean": float(batch.u_vas.mean().detach().cpu().item()),
        }

    def save(
        self,
        policy_checkpoint_path: Path,
        extra_checkpoint_path: Path,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Save Roach policy checkpoint plus separate SBA-UAS extra state."""

        # Evaluation only needs the Roach policy file. The second file captures
        # train-only state that would break the upstream checkpoint contract.
        save_roach_policy_checkpoint(
            path=policy_checkpoint_path,
            policy=self.policy,
            train_init_kwargs=self.config.train_init_kwargs,
        )
        save_sba_uas_extra_state(
            path=extra_checkpoint_path,
            critic=self.critic,
            san=self.san,
            environment_model=self.environment_model,
            reference_critic=self.reference_critic,
            reference_san=self.reference_san,
            standard_buffer=self.standard_buffer,
            familiar_buffer=self.familiar_buffer,
            critic_importance=self.critic_importance.importance(),
            san_importance=self.san_importance.importance(),
            reward_parameter_correlation=self.critic_importance,
            san_synaptic_importance=self.san_importance,
            optimizers=self._optimizer_state_dicts(),
            metadata={
                "num_timesteps": self.num_timesteps,
                **dict(metadata or {}),
            },
        )

    def load_extra_state(self, extra_checkpoint_path: Path) -> None:
        """Restore SBA-UAS extra state produced by :meth:`save`."""

        payload = load_checkpoint(extra_checkpoint_path, map_location=self.config.device)
        modules = payload.get("modules", {})
        if "critic" in modules:
            self.critic.load_state_dict(modules["critic"])
        if "san" in modules:
            self.san.load_state_dict(modules["san"])
        if "environment_model" in modules:
            self.environment_model.load_state_dict(modules["environment_model"])
        if "reference_critic" in modules:
            self.reference_critic.load_state_dict(modules["reference_critic"])
            self.reference_critic.unfreeze()
        if "reference_san" in modules:
            self.reference_san.load_state_dict(modules["reference_san"])
            self.reference_san.unfreeze()

        buffers = payload.get("buffers", {})
        if "familiar" in buffers:
            self.familiar_buffer = FamiliarExperienceBuffer.from_state_dict(
                buffers["familiar"],
                rng=self.rng,
            )
        if "standard" in buffers:
            # Restore D before B because the standard buffer owns the migration
            # link to the familiar buffer.
            self.standard_buffer = StandardReplayBuffer.from_state_dict(
                buffers["standard"],
                familiar_buffer=self.familiar_buffer,
                rng=self.rng,
            )

        trackers = payload.get("trackers", {})
        if "reward_parameter_correlation" in trackers:
            self.critic_importance.load_state_dict(
                trackers["reward_parameter_correlation"]
            )
        if "san_synaptic_importance" in trackers:
            self.san_importance.load_state_dict(trackers["san_synaptic_importance"])

        optimizers = payload.get("optimizers", {})
        self._load_optimizer_state_dicts(optimizers)
        metadata = payload.get("metadata", {})
        if "num_timesteps" in metadata:
            self.num_timesteps = int(metadata["num_timesteps"])

    def _encoded_pair(
        self,
        batch: TransitionTensorBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # The BEV model has its own objective. SAN/Critic training consumes
        # detached latents so their gradients do not update the environment model.
        self.environment_model.eval()
        with torch.no_grad():
            latent = self.environment_model.encode(batch.state)
            next_latent = self.environment_model.encode(batch.next_state)
        return latent.detach(), next_latent.detach()

    def _policy_actions_for_batch(
        self,
        batch: TransitionTensorBatch,
        next_state: bool,
    ) -> torch.Tensor:
        obs = batch.next_to_policy_obs() if next_state else batch.to_policy_obs()
        actions = self._policy_actions(
            obs,
            deterministic=True,
            clip_action=True,
        )
        return torch.as_tensor(actions, dtype=torch.float32).to(device=self.device)

    def _policy_actions(
        self,
        obs: Mapping[str, np.ndarray],
        deterministic: bool,
        clip_action: bool,
    ) -> np.ndarray:
        with torch.no_grad():
            try:
                result = self.policy.forward(
                    obs,
                    deterministic=deterministic,
                    clip_action=clip_action,
                )
            except TypeError:
                # Lightweight tests may pass a minimal fake policy that accepts
                # only the observation argument.
                result = self.policy.forward(obs)
        if isinstance(result, tuple):
            actions = result[0]
        else:
            actions = result
        return np.asarray(actions, dtype=np.float32)

    def _sample_union(self, batch_size: int) -> List[Transition]:
        return self._sample_from_buffer(
            list(self.standard_buffer) + list(self.familiar_buffer),
            batch_size,
        )

    def _sample_from_buffer(
        self,
        items: List[Transition],
        batch_size: int,
    ) -> List[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(items) < batch_size:
            return []
        return self.rng.sample(items, batch_size)

    def _clip_gradients(self, module: nn.Module) -> None:
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.config.max_grad_norm)

    def _maybe_save_checkpoint(self) -> None:
        interval = self.config.checkpoint_interval_transitions
        if interval is None or interval <= 0:
            return
        if self.num_timesteps % interval != 0:
            return
        if self.config.policy_checkpoint_path is None:
            return
        if self.config.extra_checkpoint_path is None:
            return
        self.save(
            policy_checkpoint_path=self.config.policy_checkpoint_path,
            extra_checkpoint_path=self.config.extra_checkpoint_path,
        )

    def _optimizer_state_dicts(self) -> Dict[str, Any]:
        return {
            "environment": self.environment_optimizer.state_dict(),
            "san": self.san_optimizer.state_dict(),
            "critic": self.critic_optimizer.state_dict(),
            "reference_san": self.reference_san_optimizer.state_dict(),
            "reference_critic": self.reference_critic_optimizer.state_dict(),
        }

    def _load_optimizer_state_dicts(self, optimizers: Mapping[str, Any]) -> None:
        optimizer_map = {
            "environment": self.environment_optimizer,
            "san": self.san_optimizer,
            "critic": self.critic_optimizer,
            "reference_san": self.reference_san_optimizer,
            "reference_critic": self.reference_critic_optimizer,
        }
        for name, state in optimizers.items():
            if name in optimizer_map:
                optimizer_map[name].load_state_dict(state)


def _infer_roach_shapes(env: Any) -> Tuple[Tuple[int, int, int], int, int]:
    observation_space = getattr(env, "observation_space")
    action_space = getattr(env, "action_space")
    bev_shape = tuple(int(value) for value in observation_space["birdview"].shape)
    measurement_dim = int(np.prod(observation_space["state"].shape))
    action_dim = int(np.prod(action_space.shape))
    return bev_shape, measurement_dim, action_dim


def _importance_mean(importance: Mapping[str, torch.Tensor]) -> float:
    if not importance:
        return 0.0
    values = [
        tensor.detach().float().abs().mean().cpu()
        for tensor in importance.values()
        if tensor.numel() > 0
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).mean().item())
