import torch

from sba_uas.critic.gated_critic import GatedDoubleCritic
from sba_uas.critic.san import SimilarityBasedActivationNetwork
from sba_uas.stabilization.reference_network import ReferenceNetwork
from sba_uas.stabilization.regularization import (
    make_importance_like,
    parameter_stabilization_loss,
)


def test_reference_network_is_frozen_deep_copy_of_san():
    torch.manual_seed(31)
    san = SimilarityBasedActivationNetwork(
        state_dim=3,
        measurement_dim=1,
        n_layers=2,
        units_per_layer=4,
        feature_dim=5,
        shared_hidden_units=6,
    )
    reference = ReferenceNetwork.from_model(san)

    for (name, param), (_, ref_param) in zip(
        san.named_parameters(),
        reference.model.named_parameters(),
    ):
        assert torch.allclose(param, ref_param)
        assert ref_param.requires_grad is False

        with torch.no_grad():
            param.add_(1.0)
        assert not torch.allclose(param, ref_param), name


def test_reference_network_can_sync_from_current_model_state():
    torch.manual_seed(37)
    san = SimilarityBasedActivationNetwork(
        state_dim=2,
        measurement_dim=0,
        n_layers=1,
        units_per_layer=3,
        feature_dim=4,
        shared_hidden_units=5,
    )
    reference = ReferenceNetwork.from_model(san)

    with torch.no_grad():
        for param in san.parameters():
            param.add_(0.25)
    reference.sync_from(san)

    for param, ref_param in zip(san.parameters(), reference.model.parameters()):
        assert torch.allclose(param, ref_param)
        assert ref_param.requires_grad is False


def test_parameter_stabilization_loss_penalizes_critic_drift_from_reference():
    torch.manual_seed(41)
    critic = GatedDoubleCritic(
        state_dim=3,
        action_dim=2,
        measurement_dim=1,
        hidden_units=4,
        n_layers=2,
    )
    reference = ReferenceNetwork.from_model(critic)
    importance = make_importance_like(critic, fill_value=2.0)

    zero_loss = parameter_stabilization_loss(
        model=critic,
        reference=reference,
        importance=importance,
        eta=0.5,
    )
    with torch.no_grad():
        next(critic.parameters()).add_(1.0)
    drift_loss = parameter_stabilization_loss(
        model=critic,
        reference=reference,
        importance=importance,
        eta=0.5,
    )
    drift_loss.backward()

    assert torch.allclose(zero_loss, torch.tensor(0.0))
    assert drift_loss.item() > 0.0
    assert any(param.grad is not None for param in critic.parameters())
    assert all(param.grad is None for param in reference.model.parameters())


def test_reference_and_importance_state_restore_preserves_regularization_loss():
    torch.manual_seed(43)
    critic = GatedDoubleCritic(
        state_dim=2,
        action_dim=1,
        measurement_dim=0,
        hidden_units=3,
        n_layers=1,
    )
    reference = ReferenceNetwork.from_model(critic)
    importance = make_importance_like(critic, fill_value=0.75)
    with torch.no_grad():
        next(critic.parameters()).add_(0.5)

    expected = parameter_stabilization_loss(
        model=critic,
        reference=reference,
        importance=importance,
        eta=0.25,
    )
    restored_reference = ReferenceNetwork.from_model(critic)
    restored_reference.load_state_dict(reference.state_dict())
    restored_importance = {
        name: value.clone()
        for name, value in importance.items()
    }
    actual = parameter_stabilization_loss(
        model=critic,
        reference=restored_reference,
        importance=restored_importance,
        eta=0.25,
    )

    assert torch.allclose(actual, expected)
