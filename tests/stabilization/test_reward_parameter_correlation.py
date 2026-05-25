import torch

from sba_uas.critic.gated_critic import GatedDoubleCritic
from sba_uas.stabilization.reference_network import ReferenceNetwork
from sba_uas.stabilization.regularization import parameter_stabilization_loss
from sba_uas.stabilization.reward_parameter_correlation import (
    RewardParameterCorrelation,
)


def test_reward_parameter_correlation_initial_update_only_records_baseline():
    model = torch.nn.Linear(1, 1, bias=False)
    tracker = RewardParameterCorrelation.from_model(model, damping=0.01)

    importance = tracker.update(model, batch_mean_reward=1.0)

    assert set(importance) == {"weight"}
    assert torch.allclose(importance["weight"], torch.zeros_like(model.weight))
    assert tracker.previous_reward == 1.0


def test_reward_parameter_correlation_matches_formula_after_parameter_and_reward_change():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    tracker = RewardParameterCorrelation.from_model(model, damping=0.5)
    tracker.update(model, batch_mean_reward=2.0)

    with torch.no_grad():
        model.weight.fill_(3.0)
    importance = tracker.update(model, batch_mean_reward=5.0)

    # delta_param = 2, delta_reward = 3
    # omega = 6, omega_bar = 6, Omega = 6 / (6 + 0.5)
    assert torch.allclose(importance["weight"], torch.tensor([[6.0 / 6.5]]))


def test_reward_parameter_correlation_state_restore_preserves_importance():
    model = torch.nn.Linear(1, 1, bias=False)
    tracker = RewardParameterCorrelation.from_model(model, damping=0.25)
    tracker.update(model, batch_mean_reward=1.0)
    with torch.no_grad():
        model.weight.add_(2.0)
    expected = tracker.update(model, batch_mean_reward=4.0)

    restored = RewardParameterCorrelation.from_model(model, damping=999.0)
    restored.load_state_dict(tracker.state_dict())

    assert restored.damping == 0.25
    assert torch.allclose(restored.importance()["weight"], expected["weight"])


def test_reward_parameter_correlation_importance_feeds_parameter_stabilization_loss():
    torch.manual_seed(47)
    critic = GatedDoubleCritic(
        state_dim=2,
        action_dim=1,
        measurement_dim=0,
        hidden_units=3,
        n_layers=1,
    )
    reference = ReferenceNetwork.from_model(critic)
    tracker = RewardParameterCorrelation.from_model(critic, damping=1.0e-6)
    tracker.update(critic, batch_mean_reward=0.0)

    with torch.no_grad():
        for param in critic.parameters():
            param.add_(0.1)
    importance = tracker.update(critic, batch_mean_reward=1.0)
    loss = parameter_stabilization_loss(
        model=critic,
        reference=reference,
        importance=importance,
        eta=0.5,
    )

    assert loss.item() > 0.0
