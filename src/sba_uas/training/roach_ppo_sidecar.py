"""Roach PPO training model with SBA-UAS Critic-side sidecar state.

This module is intentionally kept outside ``carla-roach/``. Roach can load
``SBAUASPPO`` through its normal Hydra ``training.entry_point`` hook, while the
Actor/policy object remains the upstream ``PpoPolicy`` and keeps the same
checkpoint contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import os
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from sba_uas.compat.roach_policy import ensure_roach_on_path
from sba_uas.training.checkpointing import save_sba_uas_extra_state
from sba_uas.training.trainer import (
    RoachCompatibleSBAUASTrainer,
    SBAUASTrainerConfig,
)


ensure_roach_on_path()

from agents.rl_birdview.models.ppo import PPO  # noqa: E402


@dataclass
class SBAUASPPOOptions:
    """Options owned by the Roach PPO integration layer."""

    enabled: bool = True
    updates_per_ppo_train: int = 1
    use_sba_critic_for_ppo_advantage: bool = True
    save_extra_state: bool = True
    extra_checkpoint_path: Optional[str] = None
    extra_checkpoint_suffix: str = "_sba_uas_extra_state"
    resume_extra_state: bool = False


_TRAINER_CONFIG_FIELDS = {item.name for item in fields(SBAUASTrainerConfig)}
_OPTION_FIELDS = {item.name for item in fields(SBAUASPPOOptions)}
_MODEL_KWARG_FIELDS = {
    "hidden_units",
    "n_layers",
    "san_feature_dim",
    "san_shared_hidden_units",
    "environment_base_channels",
    "environment_transition_hidden_dims",
    "latent_dim",
}
_PATH_FIELDS = {
    "policy_checkpoint_path",
    "extra_checkpoint_path",
}


class SBAUASPPO(PPO):
    """Drop-in Roach PPO model augmented with SBA-UAS training-side state.

    The Roach policy is still updated by PPO. SBA-UAS receives the same rollout
    stream, trains its Critic-side modules from replay, and can optionally feed
    its gated Double-Q estimates back into PPO advantage computation. That keeps
    the Actor architecture unchanged while letting the stabilized Critic guide
    policy updates after warm-up.
    """

    def __init__(
        self,
        policy,
        env,
        learning_rate: float = 1e-5,
        n_steps_total: int = 8192,
        batch_size: int = 256,
        n_epochs: int = 20,
        gamma: float = 0.99,
        gae_lambda: float = 0.9,
        clip_range: float = 0.2,
        clip_range_vf: float = None,
        ent_coef: float = 0.05,
        explore_coef: float = 0.05,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01,
        update_adv=False,
        lr_schedule_step=None,
        start_num_timesteps: int = 0,
        sba_uas: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps_total=n_steps_total,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            explore_coef=explore_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            update_adv=update_adv,
            lr_schedule_step=lr_schedule_step,
            start_num_timesteps=start_num_timesteps,
        )
        self._sba_uas_init_config = _plain_dict(sba_uas or {})
        options, trainer_config, model_kwargs = _parse_sba_uas_config(
            self._sba_uas_init_config
        )
        self.sba_uas_options = options
        self.sba_uas_trainer = None
        self.sba_uas_train_debug: Dict[str, float] = {}
        self.sba_uas_value_source = "roach_value_head"

        if not options.enabled:
            return

        trainer_config.discount = gamma
        trainer_config.train_init_kwargs = self._get_init_kwargs()
        self.sba_uas_trainer = RoachCompatibleSBAUASTrainer(
            policy=self.policy,
            env=self.env,
            config=trainer_config,
            **model_kwargs,
        )
        self.sba_uas_trainer.num_timesteps = self.num_timesteps
        extra_resume_path = _resolve_resume_extra_checkpoint_path(options)
        if extra_resume_path is not None and extra_resume_path.exists():
            self.sba_uas_trainer.load_extra_state(extra_resume_path)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        """Collect Roach PPO rollouts and mirror them into SBA-UAS buffers."""

        continue_training = super().collect_rollouts(
            env=env,
            callback=callback,
            rollout_buffer=rollout_buffer,
            n_rollout_steps=n_rollout_steps,
        )
        if continue_training and self.sba_uas_trainer is not None:
            self._ingest_rollout_buffer(rollout_buffer)
        return continue_training

    def train(self):
        """Update SBA-UAS sidecar before the normal Roach PPO update."""

        t0 = time.time()
        self.sba_uas_train_debug = self._update_sba_uas_sidecar()
        if self._should_use_sba_critic_values():
            self._rewrite_rollout_values_with_sba_critic()
            self.sba_uas_value_source = "sba_uas_gated_double_q"
        else:
            self.sba_uas_value_source = "roach_value_head"
        self.t_train_values = time.time() - t0

        result = super().train()
        if self.sba_uas_train_debug:
            self.train_debug.update(
                {
                    "sba_uas/{}".format(key): value
                    for key, value in self.sba_uas_train_debug.items()
                }
            )
        self.train_debug["sba_uas/uses_sba_critic_values"] = (
            1.0 if self.sba_uas_value_source == "sba_uas_gated_double_q" else 0.0
        )
        if self.sba_uas_trainer is not None:
            self.train_debug["sba_uas/standard_buffer_size"] = len(
                self.sba_uas_trainer.standard_buffer
            )
            self.train_debug["sba_uas/familiar_buffer_size"] = len(
                self.sba_uas_trainer.familiar_buffer
            )
        return result

    def save(self, path: str) -> None:
        """Save Roach policy checkpoint plus a separate SBA-UAS extra file."""

        super().save(path)
        if self.sba_uas_trainer is None or not self.sba_uas_options.save_extra_state:
            return
        extra_path = _resolve_extra_checkpoint_path(
            policy_path=Path(path),
            options=self.sba_uas_options,
        )
        save_sba_uas_extra_state(
            path=extra_path,
            critic=self.sba_uas_trainer.critic,
            san=self.sba_uas_trainer.san,
            environment_model=self.sba_uas_trainer.environment_model,
            reference_critic=self.sba_uas_trainer.reference_critic,
            reference_san=self.sba_uas_trainer.reference_san,
            standard_buffer=self.sba_uas_trainer.standard_buffer,
            familiar_buffer=self.sba_uas_trainer.familiar_buffer,
            critic_importance=self.sba_uas_trainer.critic_importance.importance(),
            san_importance=self.sba_uas_trainer.san_importance.importance(),
            reward_parameter_correlation=self.sba_uas_trainer.critic_importance,
            san_synaptic_importance=self.sba_uas_trainer.san_importance,
            optimizers=self.sba_uas_trainer._optimizer_state_dicts(),
            metadata={
                "policy_checkpoint": str(path),
                "num_timesteps": self.num_timesteps,
                "train_model": "{}:{}".format(
                    self.__class__.__module__,
                    self.__class__.__name__,
                ),
                "sba_uas": self._sba_uas_init_config,
            },
        )

    def _get_init_kwargs(self):
        init_kwargs = super()._get_init_kwargs()
        if self._sba_uas_init_config:
            init_kwargs["sba_uas"] = self._sba_uas_init_config
        return init_kwargs

    def _ingest_rollout_buffer(self, rollout_buffer) -> None:
        trainer = self.sba_uas_trainer
        if trainer is None:
            return

        for step_index in range(rollout_buffer.buffer_size):
            obs = {
                key: value[step_index]
                for key, value in rollout_buffer.observations.items()
            }
            next_obs = self._next_obs_from_rollout(rollout_buffer, step_index)
            dones = self._done_after_step(rollout_buffer, step_index)
            transitions = trainer.transition_adapter.annotate_step(
                obs=obs,
                actions=rollout_buffer.actions[step_index],
                rewards=rollout_buffer.rewards[step_index],
                next_obs=next_obs,
                dones=dones,
                infos=None,
                annotator=trainer.vas_annotator,
            )
            for transition in transitions:
                trainer.standard_buffer.add(transition)

        trainer.num_timesteps = self.num_timesteps

    def _next_obs_from_rollout(self, rollout_buffer, step_index: int) -> Dict[str, Any]:
        if step_index + 1 < rollout_buffer.buffer_size:
            return {
                key: value[step_index + 1]
                for key, value in rollout_buffer.observations.items()
            }
        return self._last_obs

    def _done_after_step(self, rollout_buffer, step_index: int) -> np.ndarray:
        if step_index + 1 < rollout_buffer.buffer_size:
            return rollout_buffer.dones[step_index + 1]
        return self._last_dones

    def _update_sba_uas_sidecar(self) -> Dict[str, float]:
        trainer = self.sba_uas_trainer
        if trainer is None:
            return {}
        if len(trainer.standard_buffer) < trainer.config.warmup_transitions:
            return {}

        metrics: Dict[str, float] = {}
        for _ in range(self.sba_uas_options.updates_per_ppo_train):
            metrics.update(trainer.train_step())
        trainer.last_train_metrics = metrics
        trainer.num_timesteps = self.num_timesteps
        return metrics

    def _should_use_sba_critic_values(self) -> bool:
        trainer = self.sba_uas_trainer
        if trainer is None:
            return False
        if not self.sba_uas_options.use_sba_critic_for_ppo_advantage:
            return False
        if len(trainer.standard_buffer) < trainer.config.warmup_transitions:
            return False
        return "critic/loss" in trainer.last_train_metrics

    def _rewrite_rollout_values_with_sba_critic(self) -> None:
        trainer = self.sba_uas_trainer
        if trainer is None:
            return

        for step_index in range(self.buffer.buffer_size):
            obs = {
                key: value[step_index]
                for key, value in self.buffer.observations.items()
            }
            self.buffer.values[step_index] = self._sba_critic_values(
                obs=obs,
                actions=self.buffer.actions[step_index],
            )

        next_actions = self.policy.forward(
            self._last_obs,
            deterministic=True,
            clip_action=True,
        )[0]
        last_values = self._sba_critic_values(self._last_obs, next_actions)
        self.buffer.compute_returns_and_advantage(
            last_values,
            dones=self._last_dones,
        )

    def _sba_critic_values(self, obs: Mapping[str, Any], actions: Any) -> np.ndarray:
        trainer = self.sba_uas_trainer
        if trainer is None:
            raise RuntimeError("SBA-UAS trainer is not initialized")

        trainer.environment_model.eval()
        trainer.san.eval()
        trainer.critic.eval()
        with torch.no_grad():
            batch = trainer.observation_adapter.to_tensor_batch(obs)
            action_tensor = torch.as_tensor(
                actions,
                dtype=torch.float32,
                device=trainer.device,
            )
            if action_tensor.dim() == 1:
                action_tensor = action_tensor.unsqueeze(0)
            latent = trainer.environment_model.encode(batch.birdview)
            gates, _ = trainer.san(latent, batch.measurement)
            q1, q2 = trainer.critic(
                latent,
                batch.measurement,
                action_tensor,
                gates,
            )
            values = torch.min(q1.reshape(-1), q2.reshape(-1))
        trainer.environment_model.train()
        trainer.san.train()
        trainer.critic.train()
        return values.detach().cpu().numpy().astype(np.float32)


def _parse_sba_uas_config(
    values: Mapping[str, Any],
) -> Tuple[SBAUASPPOOptions, SBAUASTrainerConfig, Dict[str, Any]]:
    option_values: Dict[str, Any] = {}
    trainer_values: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {}

    for key, value in values.items():
        if key in _OPTION_FIELDS:
            option_values[key] = value
        elif key in _TRAINER_CONFIG_FIELDS:
            trainer_values[key] = _coerce_config_value(key, value)
        elif key in _MODEL_KWARG_FIELDS:
            model_kwargs[key] = value
        else:
            raise ValueError("unknown SBA-UAS PPO option '{}'".format(key))

    if "environment_transition_hidden_dims" in model_kwargs:
        model_kwargs["environment_transition_hidden_dims"] = tuple(
            model_kwargs["environment_transition_hidden_dims"]
        )

    options = SBAUASPPOOptions(**option_values)
    if options.updates_per_ppo_train <= 0:
        raise ValueError("updates_per_ppo_train must be positive")
    trainer_config = SBAUASTrainerConfig(**trainer_values)
    return options, trainer_config, model_kwargs


def _coerce_config_value(key: str, value: Any) -> Any:
    if key in _PATH_FIELDS and value is not None:
        return Path(value)
    return value


def _resolve_extra_checkpoint_path(
    policy_path: Path,
    options: SBAUASPPOOptions,
) -> Path:
    if options.extra_checkpoint_path:
        return Path(options.extra_checkpoint_path)
    return policy_path.with_name(
        "{}{}{}".format(
            policy_path.stem,
            options.extra_checkpoint_suffix,
            policy_path.suffix,
        )
    )


def _resolve_resume_extra_checkpoint_path(
    options: SBAUASPPOOptions,
) -> Optional[Path]:
    env_path = os.environ.get("SBA_UAS_RESUME_EXTRA_STATE")
    if env_path:
        return Path(env_path)
    if options.resume_extra_state and options.extra_checkpoint_path:
        return Path(options.extra_checkpoint_path)
    return None


def _plain_dict(values: Mapping[str, Any]) -> Dict[str, Any]:
    plain = {}
    for key, value in dict(values).items():
        if isinstance(value, Path):
            plain[key] = str(value)
        elif isinstance(value, Mapping):
            plain[key] = _plain_dict(value)
        elif isinstance(value, tuple):
            plain[key] = list(value)
        else:
            plain[key] = value
    return plain


def sba_uas_config_snapshot(options: SBAUASPPOOptions) -> Dict[str, Any]:
    """Return a JSON/YAML-friendly options snapshot for docs or tooling."""

    return _plain_dict(asdict(options))
