import torch

from sba_uas.stabilization.replay_buffer import Transition
from sba_uas.stabilization.vas import ShiftedVASScorer, VASTransitionAnnotator


class FixedPredictionModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict_next_state_samples(self, state, measurement, action, num_samples):
        assert num_samples == self.predictions.shape[0]
        return self.predictions


def test_shifted_vas_averages_monte_carlo_prediction_error():
    predictions = torch.tensor(
        [
            [[1.0, 3.0]],
            [[3.0, 3.0]],
        ]
    )
    next_state = torch.tensor([[1.0, 1.0]])
    scorer = ShiftedVASScorer(monte_carlo_samples=2, sigma=2.0)

    score = scorer.score_batch(
        FixedPredictionModel(predictions),
        state=torch.zeros(1, 2),
        measurement=None,
        action=torch.zeros(1, 1),
        next_state=next_state,
    )

    assert torch.allclose(score, torch.tensor([0.75]))


def test_vas_transition_annotator_fills_u_vas_from_model_prediction_error():
    predictions = torch.tensor(
        [
            [[1.0, 3.0]],
            [[3.0, 3.0]],
        ]
    )
    annotator = VASTransitionAnnotator(
        model=FixedPredictionModel(predictions),
        scorer=ShiftedVASScorer(monte_carlo_samples=2, sigma=2.0),
    )

    annotated = annotator.annotate(
        state=torch.zeros(1, 2),
        measurement=None,
        action=torch.zeros(1, 1),
        reward=2.0,
        next_state=torch.tensor([[1.0, 1.0]]),
        next_measurement=None,
        done=True,
        metadata={"town": "Town01"},
    )

    assert isinstance(annotated, Transition)
    assert annotated.u_vas == 0.75
    assert annotated.reward == 2.0
    assert annotated.done is True
    assert annotated.metadata == {"town": "Town01"}
