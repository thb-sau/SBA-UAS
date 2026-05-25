"""Shifted VAS scoring for uncertainty-aware replay filtering."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from sba_uas.stabilization.replay_buffer import Transition


class ShiftedVASScorer:
    """Compute shifted VAS from Monte Carlo next-state predictions.

    The paper ranks transitions with ``mean(||s_hat_next - s_next||^2 / 2σ²)``.
    Lower scores indicate more familiar transitions and are therefore better
    candidates for consolidation into the familiar-experience buffer.
    """

    def __init__(self, monte_carlo_samples: int = 64, sigma: float = 1.0) -> None:
        if monte_carlo_samples <= 0:
            raise ValueError("monte_carlo_samples must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.monte_carlo_samples = monte_carlo_samples
        self.sigma = float(sigma)

    def score_batch(
        self,
        model: Any,
        state: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Return one shifted VAS score per transition in the batch."""

        predictions = model.predict_next_state_samples(
            state=state,
            measurement=measurement,
            action=action,
            num_samples=self.monte_carlo_samples,
        )
        target_next_state = next_state
        if hasattr(model, "normalize_prediction_target"):
            target_next_state = model.normalize_prediction_target(next_state)

        if predictions.dim() != target_next_state.dim() + 1:
            raise ValueError(
                "model predictions must have shape [samples, batch, ...next_state]"
            )
        if predictions.shape[1:] != target_next_state.shape:
            raise ValueError("prediction and next_state shapes do not match")

        flat_error = (predictions - target_next_state.unsqueeze(0)).pow(2)
        flat_error = flat_error.reshape(predictions.shape[0], predictions.shape[1], -1)
        per_sample_error = flat_error.sum(dim=2) / (2.0 * self.sigma * self.sigma)
        return per_sample_error.mean(dim=0)


class VASTransitionAnnotator:
    """Build replay transitions with ``u_vas`` computed from model prediction."""

    def __init__(self, model: Any, scorer: ShiftedVASScorer) -> None:
        self.model = model
        self.scorer = scorer

    def annotate(
        self,
        state: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        next_measurement: Optional[torch.Tensor],
        done: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Transition:
        """Return a single transition annotated with shifted VAS."""

        self._require_single_transition(state, action, next_state)
        if measurement is not None and measurement.shape[0] != 1:
            raise ValueError("measurement batch size must be 1")
        if next_measurement is not None and next_measurement.shape[0] != 1:
            raise ValueError("next_measurement batch size must be 1")

        with torch.no_grad():
            score = self.scorer.score_batch(
                model=self.model,
                state=state,
                measurement=measurement,
                action=action,
                next_state=next_state,
            )

        return Transition(
            state=self._first_detached(state),
            measurement=self._first_detached(measurement),
            action=self._first_detached(action),
            reward=float(reward),
            next_state=self._first_detached(next_state),
            next_measurement=self._first_detached(next_measurement),
            done=bool(done),
            u_vas=float(score.item()),
            metadata=metadata,
        )

    @staticmethod
    def _require_single_transition(
        state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> None:
        if state.shape[0] != 1:
            raise ValueError("state batch size must be 1")
        if action.shape[0] != 1:
            raise ValueError("action batch size must be 1")
        if next_state.shape[0] != 1:
            raise ValueError("next_state batch size must be 1")

    @staticmethod
    def _first_detached(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        return value.detach().cpu()[0]
