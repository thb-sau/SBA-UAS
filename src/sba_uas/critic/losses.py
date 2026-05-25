"""Losses for SAN and gated Critic training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def san_loss(
    gates: torch.Tensor,
    state_features: torch.Tensor,
    target_active_ratio: float,
    similarity_weight: float = 1.0,
    activation_weight: float = 1.0,
) -> torch.Tensor:
    """Match gate similarity to state-feature similarity and budget activity."""

    if gates.dim() != 3:
        raise ValueError("gates must have shape [batch, n_layers, units]")
    if state_features.dim() != 2:
        raise ValueError("state_features must have shape [batch, feature_dim]")
    if gates.shape[0] != state_features.shape[0]:
        raise ValueError("gates and state_features batch sizes differ")
    if not 0.0 < target_active_ratio < 1.0:
        raise ValueError("target_active_ratio must be in (0, 1)")

    gate_vectors = gates.reshape(gates.shape[0], -1)
    gate_similarity = _pairwise_cosine(gate_vectors)
    feature_similarity = _pairwise_cosine(state_features).detach()
    similarity_loss = F.mse_loss(gate_similarity, feature_similarity)

    per_transition_activity = gate_vectors.mean(dim=1)
    activation_loss = (per_transition_activity - target_active_ratio).abs().mean()
    return similarity_weight * similarity_loss + activation_weight * activation_loss


def double_q_bellman_loss(
    q1: torch.Tensor,
    q2: torch.Tensor,
    target_q: torch.Tensor,
) -> torch.Tensor:
    """Mean Bellman regression loss for twin Q estimates."""

    q1 = q1.reshape(-1)
    q2 = q2.reshape(-1)
    target_q = target_q.detach().reshape(-1)
    if q1.shape != target_q.shape or q2.shape != target_q.shape:
        raise ValueError("q1, q2, and target_q must have matching flat shapes")
    return F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)


def clipped_double_q_target(
    next_q1: torch.Tensor,
    next_q2: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    """Build a detached clipped double-Q Bellman target."""

    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be in [0, 1]")
    next_value = torch.min(next_q1.reshape(-1), next_q2.reshape(-1))
    reward = reward.reshape(-1)
    done = done.reshape(-1).to(dtype=reward.dtype)
    if reward.shape != next_value.shape or done.shape != next_value.shape:
        raise ValueError("reward, done, and next_q tensors must have matching shapes")
    return (reward + discount * (1.0 - done) * next_value).detach()


def _pairwise_cosine(values: torch.Tensor) -> torch.Tensor:
    values = F.normalize(values, p=2, dim=1, eps=1.0e-8)
    return values @ values.transpose(0, 1)
