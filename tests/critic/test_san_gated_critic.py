import torch

from sba_uas.critic.gated_critic import GatedDoubleCritic
from sba_uas.critic.losses import (
    clipped_double_q_target,
    double_q_bellman_loss,
    san_loss,
)
from sba_uas.critic.san import SimilarityBasedActivationNetwork


def test_san_outputs_sigmoid_gates_with_expected_shape_and_gradients():
    torch.manual_seed(3)
    san = SimilarityBasedActivationNetwork(
        state_dim=4,
        measurement_dim=2,
        n_layers=3,
        units_per_layer=5,
        feature_dim=7,
        shared_hidden_units=11,
        target_active_ratio=0.2,
    )

    gates, features = san(
        state=torch.randn(6, 4),
        measurement=torch.randn(6, 2),
    )
    loss = san_loss(
        gates=gates,
        state_features=features,
        target_active_ratio=0.2,
    )
    loss.backward()

    assert gates.shape == (6, 3, 5)
    assert features.shape == (6, 7)
    assert torch.all(gates > 0.0)
    assert torch.all(gates < 1.0)
    assert abs(gates.mean().item() - 0.2) < 0.08
    assert any(param.grad is not None for param in san.parameters())


def test_gated_double_critic_consumes_san_gates_and_backpropagates_bellman_loss():
    torch.manual_seed(5)
    san = SimilarityBasedActivationNetwork(
        state_dim=4,
        measurement_dim=2,
        n_layers=2,
        units_per_layer=8,
        feature_dim=6,
        shared_hidden_units=10,
        target_active_ratio=0.3,
    )
    critic = GatedDoubleCritic(
        state_dim=4,
        action_dim=2,
        measurement_dim=2,
        hidden_units=8,
        n_layers=2,
    )

    state = torch.randn(5, 4)
    measurement = torch.randn(5, 2)
    action = torch.randn(5, 2)
    gates, _ = san(state, measurement)
    q1, q2 = critic(state, measurement, action, gates)
    target_q = torch.randn(5)
    loss = double_q_bellman_loss(q1, q2, target_q)
    loss.backward()

    assert q1.shape == (5,)
    assert q2.shape == (5,)
    assert loss.item() >= 0.0
    assert any(param.grad is not None for param in critic.parameters())
    assert any(param.grad is not None for param in san.parameters())


def test_clipped_double_q_target_uses_lower_next_q_and_done_mask():
    target = clipped_double_q_target(
        next_q1=torch.tensor([10.0, 2.0]),
        next_q2=torch.tensor([4.0, 8.0]),
        reward=torch.tensor([1.0, 5.0]),
        done=torch.tensor([0.0, 1.0]),
        discount=0.5,
    )

    assert torch.allclose(target, torch.tensor([3.0, 5.0]))
