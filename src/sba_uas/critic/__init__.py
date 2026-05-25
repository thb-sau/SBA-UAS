"""Critic-side SBA-UAS components."""

from sba_uas.critic.gated_critic import GatedDoubleCritic, GatedQNetwork
from sba_uas.critic.losses import (
    clipped_double_q_target,
    double_q_bellman_loss,
    san_loss,
)
from sba_uas.critic.san import SimilarityBasedActivationNetwork


__all__ = [
    "GatedDoubleCritic",
    "GatedQNetwork",
    "SimilarityBasedActivationNetwork",
    "clipped_double_q_target",
    "double_q_bellman_loss",
    "san_loss",
]
