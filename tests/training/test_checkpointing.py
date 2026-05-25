import torch

from sba_uas.stabilization.replay_buffer import (
    FamiliarExperienceBuffer,
    StandardReplayBuffer,
    Transition,
)
from sba_uas.stabilization.synaptic_importance import (
    SynapticIntelligenceImportance,
)
from sba_uas.training.checkpointing import (
    load_checkpoint,
    save_roach_policy_checkpoint,
    save_sba_uas_extra_state,
)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def get_init_kwargs(self):
        return {
            "policy_head_arch": [256, 256],
            "value_head_arch": [256, 256],
        }


def transition(label, u_vas):
    return Transition(
        state=label,
        measurement=None,
        action="action-{}".format(label),
        reward=1.0,
        next_state="next-{}".format(label),
        next_measurement=None,
        done=False,
        u_vas=u_vas,
    )


def test_save_roach_policy_checkpoint_uses_roach_contract_keys(tmp_path):
    path = tmp_path / "policy.pth"
    policy = TinyPolicy()

    save_roach_policy_checkpoint(
        path=path,
        policy=policy,
        train_init_kwargs={"learning_rate": 1.0e-5},
    )

    payload = load_checkpoint(path)
    assert set(payload) == {
        "policy_state_dict",
        "policy_init_kwargs",
        "train_init_kwargs",
    }
    assert payload["policy_init_kwargs"] == policy.get_init_kwargs()
    assert payload["train_init_kwargs"] == {"learning_rate": 1.0e-5}
    assert "layer.weight" in payload["policy_state_dict"]


def test_save_sba_uas_extra_state_keeps_modules_buffers_and_importance(tmp_path):
    path = tmp_path / "sba_uas_extra_state.pth"
    critic = torch.nn.Linear(2, 1)
    san = torch.nn.Linear(2, 3)
    san_tracker = SynapticIntelligenceImportance.from_model(san)
    familiar = FamiliarExperienceBuffer(capacity=2)
    standard = StandardReplayBuffer(capacity=2, familiar_buffer=familiar)
    standard.add(transition("standard", 0.5))
    familiar.add(transition("familiar", 0.1))
    importance = {"weight": torch.ones_like(critic.weight)}

    save_sba_uas_extra_state(
        path=path,
        critic=critic,
        san=san,
        standard_buffer=standard,
        familiar_buffer=familiar,
        critic_importance=importance,
        san_synaptic_importance=san_tracker,
        metadata={"stage": "smoke"},
    )

    payload = load_checkpoint(path)
    assert payload["metadata"] == {"stage": "smoke"}
    assert set(payload["modules"]) == {"critic", "san"}
    assert set(payload["buffers"]) == {"standard", "familiar"}
    assert set(payload["trackers"]) == {"san_synaptic_importance"}
    assert torch.allclose(payload["importance"]["critic"]["weight"], importance["weight"])
    assert payload["buffers"]["standard"]["items"][0].state == "standard"
    assert payload["buffers"]["familiar"]["items"][0].state == "familiar"
