from collections import deque
from pathlib import Path

import gym
import numpy as np
import torch

from sba_uas.training.checkpointing import load_checkpoint
from sba_uas.training.roach_ppo_sidecar import SBAUASPPO


class FakeRoachVecEnv:
    def __init__(self):
        self.num_envs = 2
        self.observation_space = gym.spaces.Dict(
            {
                "birdview": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, 32, 32),
                    dtype=np.uint8,
                ),
                "state": gym.spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(6,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.step_index = 0

    def seed(self, seed):
        self._seed = seed

    def reset(self):
        self.step_index = 0
        return self._obs()

    def step(self, actions):
        self.step_index += 1
        actions = np.asarray(actions, dtype=np.float32)
        obs = self._obs()
        rewards = 1.0 - np.sum(actions * actions, axis=1)
        dones = np.array([False, self.step_index % 2 == 0])
        infos = [self._info(), self._info()]
        return obs, rewards.astype(np.float32), dones, infos

    def _obs(self):
        birdview = np.zeros((2, 3, 32, 32), dtype=np.uint8)
        birdview[0].fill(self.step_index)
        birdview[1].fill(self.step_index + 1)
        state = np.array(
            [
                [self.step_index, 0.0, 0.1, 0.2, 0.3, 0.4],
                [self.step_index, 1.0, 1.1, 1.2, 1.3, 1.4],
            ],
            dtype=np.float32,
        )
        return {"birdview": birdview, "state": state}

    @staticmethod
    def _info():
        return {
            "episode_stat": {"driving_score": 1.0},
            "reward_debug": {"debug_texts": []},
            "terminal_debug": {
                "debug_texts": [],
                "exploration_suggest": {
                    "n_steps": 0,
                    "suggest": ("", ""),
                },
            },
        }


class FakeActionDistribution:
    def proba_distribution(self, mu, sigma):
        return FakeDistributionWrapper(mu, sigma)


class FakeDistributionWrapper:
    def __init__(self, mu, sigma):
        self.distribution = torch.distributions.Independent(
            torch.distributions.Normal(mu, sigma),
            1,
        )


class FakeRoachPolicy(torch.nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_dist = FakeActionDistribution()
        self.policy_head = torch.nn.Linear(6, 2)
        self.value_head = torch.nn.Linear(6, 1)
        self.log_std = torch.nn.Parameter(torch.zeros(2))
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1.0e-3)

    def forward(self, obs, deterministic=False, clip_action=False):
        with torch.no_grad():
            state = torch.as_tensor(
                obs["state"],
                dtype=torch.float32,
                device=self.device,
            )
            mu, sigma, distribution = self._distribution(state)
            actions = mu if deterministic else distribution.rsample()
            actions = actions.clamp(-1.0, 1.0)
            values = self.value_head(state).flatten()
            log_prob = distribution.log_prob(actions)
        actions_np = actions.cpu().numpy().astype(np.float32)
        if clip_action:
            actions_np = np.clip(
                actions_np,
                self.action_space.low,
                self.action_space.high,
            )
        features = state.cpu().numpy().astype(np.float32)
        return (
            actions_np,
            values.cpu().numpy(),
            log_prob.cpu().numpy(),
            mu.cpu().numpy(),
            sigma.cpu().numpy(),
            features,
        )

    def forward_value(self, obs):
        with torch.no_grad():
            state = torch.as_tensor(
                obs["state"],
                dtype=torch.float32,
                device=self.device,
            )
            values = self.value_head(state).flatten()
        return values.cpu().numpy()

    def evaluate_actions(self, obs_dict, actions, exploration_suggests, detach_values=False):
        state = obs_dict["state"].float()
        values = self.value_head(state.detach() if detach_values else state).flatten()
        mu, sigma, distribution = self._distribution(state)
        log_prob = distribution.log_prob(actions)
        entropy_loss = -distribution.entropy().mean()
        exploration_loss = values.new_tensor(0.0)
        return values, log_prob, entropy_loss, exploration_loss, distribution

    def get_init_kwargs(self):
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy_head_arch": [256, 256],
            "value_head_arch": [256, 256],
        }

    def _distribution(self, state):
        mu = torch.tanh(self.policy_head(state))
        sigma = torch.nn.functional.softplus(self.log_std).expand_as(mu) + 0.1
        distribution = torch.distributions.Independent(
            torch.distributions.Normal(mu, sigma),
            1,
        )
        return mu, sigma, distribution


class NoopCallback:
    def on_step(self):
        return True


def _small_sba_uas_config():
    return {
        "standard_buffer_capacity": 2,
        "familiar_buffer_capacity": 4,
        "batch_size": 2,
        "environment_batch_size": 2,
        "reference_batch_size": 1,
        "warmup_transitions": 2,
        "shifted_vas_samples": 1,
        "environment_prediction_samples": 1,
        "learning_rate": 1.0e-3,
        "reference_learning_rate": 1.0e-3,
        "environment_learning_rate": 1.0e-3,
        "max_grad_norm": 5.0,
        "updates_per_ppo_train": 1,
        "hidden_units": 4,
        "n_layers": 1,
        "latent_dim": 4,
        "san_feature_dim": 4,
        "san_shared_hidden_units": 8,
        "environment_base_channels": 1,
        "environment_transition_hidden_dims": [8],
    }


def _build_model(env, policy, sba_uas_config=None):
    return SBAUASPPO(
        policy=policy,
        env=env,
        learning_rate=1.0e-3,
        n_steps_total=2,
        batch_size=2,
        n_epochs=1,
        gamma=0.9,
        gae_lambda=0.9,
        ent_coef=0.0,
        explore_coef=0.0,
        vf_coef=0.5,
        target_kl=None,
        sba_uas=sba_uas_config or _small_sba_uas_config(),
    )


def test_sba_uas_ppo_trains_policy_and_saves_split_extra_state(tmp_path):
    env = FakeRoachVecEnv()
    policy = FakeRoachPolicy(env.observation_space, env.action_space)
    model = _build_model(env, policy)
    model.ep_stat_buffer = deque(maxlen=100)
    model._last_obs = env.reset()
    model._last_dones = np.zeros((env.num_envs,), dtype=np.bool_)
    initial_state = {
        name: param.detach().clone()
        for name, param in policy.named_parameters()
    }

    assert model.collect_rollouts(env, NoopCallback(), model.buffer, model.n_steps)
    model.train()

    assert len(model.sba_uas_trainer.standard_buffer) == 2
    assert "sba_uas/critic/loss" in model.train_debug
    assert model.train_debug["sba_uas/uses_sba_critic_values"] == 1.0
    assert any(
        not torch.allclose(initial_state[name], param)
        for name, param in policy.named_parameters()
    )

    policy_path = tmp_path / "ckpt_2.pth"
    model.save(policy_path.as_posix())
    extra_path = tmp_path / "ckpt_2_sba_uas_extra_state.pth"

    policy_payload = load_checkpoint(policy_path)
    extra_payload = load_checkpoint(extra_path)
    assert "sba_uas" in policy_payload["train_init_kwargs"]
    assert "critic" in extra_payload["modules"]
    assert "standard" in extra_payload["buffers"]
    assert extra_payload["metadata"]["num_timesteps"] == model.num_timesteps


def test_sba_uas_ppo_can_resume_extra_state_from_environment(tmp_path, monkeypatch):
    env = FakeRoachVecEnv()
    model = _build_model(
        env,
        FakeRoachPolicy(env.observation_space, env.action_space),
    )
    model.ep_stat_buffer = deque(maxlen=100)
    model._last_obs = env.reset()
    model._last_dones = np.zeros((env.num_envs,), dtype=np.bool_)
    assert model.collect_rollouts(env, NoopCallback(), model.buffer, model.n_steps)
    model.train()
    policy_path = tmp_path / "ckpt_2.pth"
    model.save(policy_path.as_posix())

    extra_path = tmp_path / "ckpt_2_sba_uas_extra_state.pth"
    monkeypatch.setenv("SBA_UAS_RESUME_EXTRA_STATE", extra_path.as_posix())
    restored = _build_model(
        env,
        FakeRoachPolicy(env.observation_space, env.action_space),
    )

    assert len(restored.sba_uas_trainer.standard_buffer) == len(
        model.sba_uas_trainer.standard_buffer
    )
    assert restored.sba_uas_trainer.num_timesteps == model.num_timesteps
