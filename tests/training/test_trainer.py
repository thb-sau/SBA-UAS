import numpy as np
import torch

from sba_uas.stabilization.environment_model import BayesianBEVEnvironmentModel
from sba_uas.training.checkpointing import load_checkpoint
from sba_uas.training.trainer import (
    RoachCompatibleSBAUASTrainer,
    SBAUASTrainerConfig,
)


class FakeSpace:
    def __init__(self, shape):
        self.shape = shape

    def seed(self, seed):
        self._seed = seed


class FakeRoachVecEnv:
    def __init__(self):
        self.num_envs = 2
        self.observation_space = {
            "birdview": FakeSpace((3, 32, 32)),
            "state": FakeSpace((6,)),
        }
        self.action_space = FakeSpace((2,))
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
        dones = np.array([False, self.step_index % 3 == 0])
        infos = [
            {"town": "Town01", "task_idx": 0},
            {"town": "Town02", "task_idx": 1},
        ]
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


class FakeRoachPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.bias = torch.nn.Parameter(torch.zeros(2))

    def forward(self, obs, deterministic=False, clip_action=False):
        state = np.asarray(obs["state"], dtype=np.float32)
        bias = self.bias.detach().cpu().numpy()
        actions = np.tanh(state[:, :2] * 0.1 + bias)
        values = np.zeros((state.shape[0],), dtype=np.float32)
        log_probs = np.zeros((state.shape[0],), dtype=np.float32)
        mu = actions.copy()
        sigma = np.ones_like(actions)
        features = state.copy()
        return actions.astype(np.float32), values, log_probs, mu, sigma, features

    def get_init_kwargs(self):
        return {
            "policy_head_arch": [256, 256],
            "value_head_arch": [256, 256],
        }


def build_trainer(tmp_path):
    config = SBAUASTrainerConfig(
        device="cpu",
        standard_buffer_capacity=2,
        familiar_buffer_capacity=4,
        batch_size=2,
        environment_batch_size=2,
        reference_batch_size=1,
        warmup_transitions=2,
        updates_per_env_step=1,
        shifted_vas_samples=1,
        environment_prediction_samples=1,
        learning_rate=1.0e-3,
        reference_learning_rate=1.0e-3,
        environment_learning_rate=1.0e-3,
        max_grad_norm=5.0,
        train_init_kwargs={"learning_rate": 1.0e-5},
    )
    env_model = BayesianBEVEnvironmentModel(
        bev_shape=(3, 32, 32),
        action_dim=2,
        measurement_dim=6,
        latent_dim=4,
        base_channels=1,
        transition_hidden_dims=(8,),
    )
    return RoachCompatibleSBAUASTrainer(
        policy=FakeRoachPolicy(),
        env=FakeRoachVecEnv(),
        config=config,
        environment_model=env_model,
        hidden_units=4,
        n_layers=1,
        san_feature_dim=4,
        san_shared_hidden_units=8,
        latent_dim=4,
    )


def test_roach_compatible_trainer_collects_updates_and_saves_split_checkpoints(tmp_path):
    trainer = build_trainer(tmp_path)

    trainer.learn(total_timesteps=6, seed=123)

    assert trainer.num_timesteps >= 6
    assert len(trainer.standard_buffer) == 2
    assert len(trainer.familiar_buffer) > 0
    assert "critic/loss" in trainer.last_train_metrics
    assert "env_model/loss" in trainer.last_train_metrics

    policy_path = tmp_path / "policy.pth"
    extra_path = tmp_path / "sba_uas_extra_state.pth"
    trainer.save(policy_path, extra_path, metadata={"test": "smoke"})

    policy_payload = load_checkpoint(policy_path)
    extra_payload = load_checkpoint(extra_path)
    assert set(policy_payload) == {
        "policy_state_dict",
        "policy_init_kwargs",
        "train_init_kwargs",
    }
    assert "environment_model" in extra_payload["modules"]
    assert "critic" in extra_payload["modules"]
    assert "san" in extra_payload["modules"]
    assert "standard" in extra_payload["buffers"]
    assert "familiar" in extra_payload["buffers"]
    assert "reward_parameter_correlation" in extra_payload["trackers"]
    assert "san_synaptic_importance" in extra_payload["trackers"]
    assert "environment" in extra_payload["optimizers"]
    assert extra_payload["metadata"]["test"] == "smoke"


def test_roach_compatible_trainer_can_restore_extra_state(tmp_path):
    trainer = build_trainer(tmp_path)
    trainer.learn(total_timesteps=4, seed=123)
    extra_path = tmp_path / "sba_uas_extra_state.pth"
    trainer.save(tmp_path / "policy.pth", extra_path)

    restored = build_trainer(tmp_path)
    restored.load_extra_state(extra_path)

    assert restored.num_timesteps == trainer.num_timesteps
    assert len(restored.standard_buffer) == len(trainer.standard_buffer)
    assert len(restored.familiar_buffer) == len(trainer.familiar_buffer)
