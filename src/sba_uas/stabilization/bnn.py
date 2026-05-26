"""Small Bayesian neural-network building blocks for SBA-UAS dynamics."""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
from torch import nn
from torch.nn import functional as F


class BayesianLinear(nn.Module):
    """Linear layer with a factorized Gaussian posterior over parameters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        posterior_rho_init: float = -5.0,
    ) -> None:
        super().__init__()
        if prior_std <= 0:
            raise ValueError("prior_std must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = float(prior_std)

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters(posterior_rho_init)

    def reset_parameters(self, posterior_rho_init: float) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_rho, posterior_rho_init)
        nn.init.constant_(self.bias_rho, posterior_rho_init)

    def forward(self, inputs: torch.Tensor, sample: bool = True) -> torch.Tensor:
        weight = self._sample(self.weight_mu, self.weight_rho) if sample else self.weight_mu
        bias = self._sample(self.bias_mu, self.bias_rho) if sample else self.bias_mu
        return F.linear(inputs, weight, bias)

    def kl_loss(self) -> torch.Tensor:
        return self._normal_kl(self.weight_mu, self.weight_rho) + self._normal_kl(
            self.bias_mu, self.bias_rho
        )

    @staticmethod
    def _sample(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        # Softplus keeps posterior std positive while allowing unconstrained rho
        # parameters during optimization.
        sigma = F.softplus(rho)
        return mu + sigma * torch.randn_like(mu)

    def _normal_kl(self, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(rho)
        prior_std = torch.as_tensor(self.prior_std, dtype=mu.dtype, device=mu.device)
        return (
            torch.log(prior_std / sigma)
            + (sigma.pow(2) + mu.pow(2)) / (2.0 * prior_std.pow(2))
            - 0.5
        ).sum()


class BayesianMLP(nn.Module):
    """MLP composed of BayesianLinear layers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        prior_std: float = 1.0,
    ) -> None:
        super().__init__()
        dims: List[int] = [input_dim] + list(hidden_dims) + [output_dim]
        self.layers = nn.ModuleList(
            [
                BayesianLinear(dims[index], dims[index + 1], prior_std=prior_std)
                for index in range(len(dims) - 1)
            ]
        )

    def forward(self, inputs: torch.Tensor, sample: bool = True) -> torch.Tensor:
        hidden = inputs
        for layer in self.layers[:-1]:
            hidden = F.relu(layer(hidden, sample=sample))
        return self.layers[-1](hidden, sample=sample)

    def kl_loss(self) -> torch.Tensor:
        total = None
        for layer in self.layers:
            value = layer.kl_loss()
            total = value if total is None else total + value
        if total is None:
            raise RuntimeError("BayesianMLP has no layers")
        return total
