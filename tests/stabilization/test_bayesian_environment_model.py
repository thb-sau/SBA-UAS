import torch

from sba_uas.stabilization.environment_model import (
    BayesianBEVEnvironmentModel,
    BayesianDynamicsModel,
)
from sba_uas.stabilization.replay_buffer import StandardReplayBuffer
from sba_uas.stabilization.vas import ShiftedVASScorer, VASTransitionAnnotator


def test_bev_environment_model_matches_roach_like_birdview_contract():
    torch.manual_seed(101)
    model = BayesianBEVEnvironmentModel(
        bev_shape=(15, 64, 64),
        action_dim=2,
        measurement_dim=6,
        latent_dim=16,
        base_channels=4,
        transition_hidden_dims=(32,),
    )
    birdview = torch.randint(0, 256, (2, 15, 64, 64), dtype=torch.uint8)
    measurement = torch.randn(2, 6)
    action = torch.randn(2, 2)

    latent = model.encode(birdview)
    reconstruction = model.reconstruct_state(birdview)
    samples = model.predict_next_state_samples(
        state=birdview,
        measurement=measurement,
        action=action,
        num_samples=3,
    )

    assert latent.shape == (2, 16)
    assert reconstruction.shape == (2, 15, 64, 64)
    assert samples.shape == (3, 2, 15, 64, 64)
    assert torch.all(samples >= 0.0)
    assert torch.all(samples <= 1.0)


def test_bev_environment_model_composite_loss_backpropagates():
    torch.manual_seed(103)
    model = BayesianBEVEnvironmentModel(
        bev_shape=(15, 64, 64),
        action_dim=2,
        measurement_dim=6,
        latent_dim=8,
        base_channels=2,
        transition_hidden_dims=(16,),
    )
    state = torch.randint(0, 256, (2, 15, 64, 64), dtype=torch.uint8)
    next_state = torch.randint(0, 256, (2, 15, 64, 64), dtype=torch.uint8)
    measurement = torch.randn(2, 6)
    action = torch.randn(2, 2)

    terms = model.loss_terms(
        state=state,
        measurement=measurement,
        action=action,
        next_state=next_state,
        prediction_samples=2,
    )
    loss = model.training_loss(
        state=state,
        measurement=measurement,
        action=action,
        next_state=next_state,
        kl_weight=1.0e-6,
        prediction_samples=2,
    )
    loss.backward()

    assert set(terms) == {
        "state_reconstruction",
        "next_state_reconstruction",
        "latent_prediction",
        "observation_prediction",
        "kl",
    }
    assert loss.item() >= 0.0
    assert any(param.grad is not None for param in model.parameters())


def test_bev_environment_model_normalizes_vas_target_from_raw_roach_masks():
    torch.manual_seed(107)
    model = BayesianBEVEnvironmentModel(
        bev_shape=(15, 64, 64),
        action_dim=2,
        measurement_dim=6,
        latent_dim=8,
        base_channels=2,
        transition_hidden_dims=(16,),
    )
    scorer = ShiftedVASScorer(monte_carlo_samples=2, sigma=1.0)
    state = torch.randint(0, 256, (1, 15, 64, 64), dtype=torch.uint8)
    next_state = torch.randint(0, 256, (1, 15, 64, 64), dtype=torch.uint8)

    score = scorer.score_batch(
        model=model,
        state=state,
        measurement=torch.randn(1, 6),
        action=torch.randn(1, 2),
        next_state=next_state,
    )

    assert score.shape == (1,)
    assert score.item() >= 0.0


def test_bayesian_dynamics_model_predicts_monte_carlo_next_state_samples():
    torch.manual_seed(7)
    model = BayesianDynamicsModel(
        state_dim=2,
        action_dim=1,
        measurement_dim=1,
        hidden_dims=(8,),
    )

    samples = model.predict_next_state_samples(
        state=torch.zeros(3, 2),
        measurement=torch.ones(3, 1),
        action=torch.zeros(3, 1),
        num_samples=5,
    )

    assert samples.shape == (5, 3, 2)
    assert model.kl_loss().item() >= 0.0


def test_training_dynamics_model_reduces_shifted_vas_for_seen_transition():
    torch.manual_seed(11)
    model = BayesianDynamicsModel(
        state_dim=2,
        action_dim=1,
        measurement_dim=1,
        hidden_dims=(16,),
    )
    scorer = ShiftedVASScorer(monte_carlo_samples=8, sigma=1.0)
    state = torch.tensor([[0.2, -0.1]]).repeat(16, 1)
    measurement = torch.tensor([[0.5]]).repeat(16, 1)
    action = torch.tensor([[0.3]]).repeat(16, 1)
    next_state = torch.tensor([[0.7, -0.4]]).repeat(16, 1)

    before = scorer.score_batch(model, state, measurement, action, next_state).mean()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(120):
        optimizer.zero_grad()
        loss = model.training_loss(
            state=state,
            measurement=measurement,
            action=action,
            next_state=next_state,
            kl_weight=0.0,
        )
        loss.backward()
        optimizer.step()

    after = scorer.score_batch(model, state, measurement, action, next_state).mean()

    assert after.item() < before.item() * 0.2


def test_bayesian_model_annotates_transition_before_replay_storage():
    torch.manual_seed(23)
    model = BayesianDynamicsModel(
        state_dim=2,
        action_dim=1,
        measurement_dim=0,
        hidden_dims=(8,),
    )
    annotator = VASTransitionAnnotator(
        model=model,
        scorer=ShiftedVASScorer(monte_carlo_samples=4, sigma=1.0),
    )
    standard = StandardReplayBuffer(capacity=2)

    annotated = annotator.annotate(
        state=torch.tensor([[0.1, 0.2]]),
        measurement=None,
        action=torch.tensor([[0.3]]),
        reward=1.5,
        next_state=torch.tensor([[0.4, 0.5]]),
        next_measurement=None,
        done=False,
    )
    standard.add(annotated)

    stored = next(iter(standard))
    assert stored.u_vas >= 0.0
    assert stored.u_vas == annotated.u_vas
