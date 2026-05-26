"""Bayesian environment models used to produce shifted VAS scores."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from sba_uas.stabilization.bnn import BayesianMLP


class ResidualConvBlock(nn.Module):
    """Small ResNet-style block used by the BEV encoder and decoder."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
            )
        else:
            self.skip = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        hidden = F.relu(self.conv1(inputs))
        hidden = self.conv2(hidden)
        return F.relu(hidden + residual)


class RoachBEVEncoder(nn.Module):
    """Encode Roach birdview masks into a compact latent vector."""

    def __init__(
        self,
        bev_shape: Tuple[int, int, int],
        latent_dim: int = 256,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        channels, height, width = _validate_bev_shape(bev_shape)
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        self.bev_shape = (channels, height, width)
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        # Five stride-2 stages downsample Roach's 192x192 BEV to 6x6. Keeping
        # this explicit makes shape mismatches fail early in _validate_bev_shape.
        self.feature_shape = (
            base_channels * 8,
            height // 32,
            width // 32,
        )
        flat_dim = (
            self.feature_shape[0] * self.feature_shape[1] * self.feature_shape[2]
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            ResidualConvBlock(base_channels, base_channels),
            ResidualConvBlock(base_channels, base_channels * 2, stride=2),
            ResidualConvBlock(base_channels * 2, base_channels * 4, stride=2),
            ResidualConvBlock(base_channels * 4, base_channels * 8, stride=2),
            ResidualConvBlock(base_channels * 8, base_channels * 8, stride=2),
            nn.Flatten(),
        )
        self.to_latent = nn.Linear(flat_dim, latent_dim)

    def forward(self, birdview: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.cnn(birdview))


class RoachBEVDecoder(nn.Module):
    """Decode latent vectors back into normalized Roach birdview masks."""

    def __init__(
        self,
        bev_shape: Tuple[int, int, int],
        latent_dim: int = 256,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        channels, height, width = _validate_bev_shape(bev_shape)
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        self.bev_shape = (channels, height, width)
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.feature_shape = (
            base_channels * 8,
            height // 32,
            width // 32,
        )
        flat_dim = (
            self.feature_shape[0] * self.feature_shape[1] * self.feature_shape[2]
        )

        self.from_latent = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 8,
                base_channels * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            ResidualConvBlock(base_channels * 8, base_channels * 8),
            nn.ConvTranspose2d(
                base_channels * 8,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            ResidualConvBlock(base_channels * 4, base_channels * 4),
            nn.ConvTranspose2d(
                base_channels * 4,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            ResidualConvBlock(base_channels * 2, base_channels * 2),
            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            ResidualConvBlock(base_channels, base_channels),
            nn.ConvTranspose2d(
                base_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(base_channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.from_latent(latent)
        hidden = hidden.view(latent.shape[0], *self.feature_shape)
        return self.decoder(hidden)


class BayesianBEVEnvironmentModel(nn.Module):
    """Paper-style BEV autoencoder plus Bayesian latent transition model.

    The model follows the Roach observation contract: ``birdview`` is a
    channel-first mask tensor from ``RlBirdviewWrapper`` and ``measurement`` is
    the wrapper's ``state`` vector. Inputs may be raw Roach uint8 masks in
    ``[0, 255]``; the model normalizes them internally and all decoded
    predictions are returned in normalized ``[0, 1]`` space.
    """

    def __init__(
        self,
        bev_shape: Tuple[int, int, int] = (15, 192, 192),
        action_dim: int = 2,
        measurement_dim: int = 6,
        latent_dim: int = 256,
        base_channels: int = 32,
        transition_hidden_dims: Iterable[int] = (512, 512),
        prior_std: float = 1.0,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if measurement_dim < 0:
            raise ValueError("measurement_dim cannot be negative")

        self.bev_shape = _validate_bev_shape(bev_shape)
        self.action_dim = action_dim
        self.measurement_dim = measurement_dim
        self.latent_dim = latent_dim
        self.encoder = RoachBEVEncoder(
            bev_shape=self.bev_shape,
            latent_dim=latent_dim,
            base_channels=base_channels,
        )
        self.decoder = RoachBEVDecoder(
            bev_shape=self.bev_shape,
            latent_dim=latent_dim,
            base_channels=base_channels,
        )
        self.transition_model = BayesianMLP(
            input_dim=latent_dim + measurement_dim + action_dim,
            output_dim=latent_dim,
            hidden_dims=transition_hidden_dims,
            prior_std=prior_std,
        )

    def encode(self, birdview: torch.Tensor) -> torch.Tensor:
        """Return latent ``z = ENC(s)`` for Roach BEV masks."""

        return self.encoder(self.normalize_birdview(birdview))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Return normalized BEV reconstruction ``DEC(z)``."""

        self._check_latent(latent)
        return self.decoder(latent)

    def forward(
        self,
        state: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """Predict normalized next BEV masks from ``state, measurement, action``."""

        latent = self.encode(state)
        next_latent = self.predict_next_latent(
            latent=latent,
            measurement=measurement,
            action=action,
            sample=sample,
        )
        return self.decode(next_latent)

    def reconstruct_state(self, state: torch.Tensor) -> torch.Tensor:
        """Autoencode the current BEV state in normalized mask space."""

        return self.decode(self.encode(state))

    def predict_next_latent(
        self,
        latent: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """Predict ``z'_hat = BNN(z, m, a)``."""

        return self.transition_model(
            self._compose_transition_input(
                latent=latent,
                measurement=measurement,
                action=action,
            ),
            sample=sample,
        )

    def predict_next_latent_samples(
        self,
        state: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Return Monte Carlo latent predictions with shape ``[N, B, Z]``."""

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        latent = self.encode(state)
        samples = [
            self.predict_next_latent(
                latent=latent,
                measurement=measurement,
                action=action,
                sample=True,
            )
            for _ in range(num_samples)
        ]
        return torch.stack(samples, dim=0)

    def predict_next_state_samples(
        self,
        state: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Return Monte Carlo next-BEV predictions with shape ``[N, B, C, H, W]``."""

        latent_samples = self.predict_next_latent_samples(
            state=state,
            measurement=measurement,
            action=action,
            num_samples=num_samples,
        )
        decoded = [self.decode(latent_samples[index]) for index in range(num_samples)]
        return torch.stack(decoded, dim=0)

    def loss_terms(
        self,
        state: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
        next_state: torch.Tensor,
        prediction_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Return the reconstruction and transition losses from Eq. (9)."""

        if prediction_samples <= 0:
            raise ValueError("prediction_samples must be positive")

        state_target = self.normalize_birdview(state)
        next_state_target = self.normalize_birdview(next_state)
        latent = self.encoder(state_target)
        next_latent = self.encoder(next_state_target)

        # Both current and next states are reconstructed so the latent space
        # remains useful for present-state gating and transition prediction.
        state_reconstruction = self.decode(latent)
        next_state_reconstruction = self.decode(next_latent)
        state_reconstruction_loss = F.mse_loss(state_reconstruction, state_target)
        next_state_reconstruction_loss = F.mse_loss(
            next_state_reconstruction,
            next_state_target,
        )

        latent_prediction_losses = []
        observation_prediction_losses = []
        # Average MC transition losses because each Bayesian sample is a valid
        # posterior draw, not a separate supervised target.
        for _ in range(prediction_samples):
            predicted_next_latent = self.predict_next_latent(
                latent=latent,
                measurement=measurement,
                action=action,
                sample=True,
            )
            predicted_next_state = self.decode(predicted_next_latent)
            latent_prediction_losses.append(
                F.mse_loss(predicted_next_latent, next_latent)
            )
            observation_prediction_losses.append(
                F.mse_loss(predicted_next_state, next_state_target)
            )

        latent_prediction_loss = torch.stack(latent_prediction_losses).mean()
        observation_prediction_loss = torch.stack(
            observation_prediction_losses
        ).mean()

        return {
            "state_reconstruction": state_reconstruction_loss,
            "next_state_reconstruction": next_state_reconstruction_loss,
            "latent_prediction": latent_prediction_loss,
            "observation_prediction": observation_prediction_loss,
            "kl": self.kl_loss(),
        }

    def training_loss(
        self,
        state: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
        next_state: torch.Tensor,
        kl_weight: float = 1.0e-4,
        reconstruction_weight: float = 1.0,
        latent_prediction_weight: float = 1.0,
        observation_prediction_weight: float = 1.0,
        prediction_samples: int = 1,
    ) -> torch.Tensor:
        """Return the weighted paper environment-model objective."""

        if kl_weight < 0.0:
            raise ValueError("kl_weight must be non-negative")
        if reconstruction_weight < 0.0:
            raise ValueError("reconstruction_weight must be non-negative")
        if latent_prediction_weight < 0.0:
            raise ValueError("latent_prediction_weight must be non-negative")
        if observation_prediction_weight < 0.0:
            raise ValueError("observation_prediction_weight must be non-negative")

        terms = self.loss_terms(
            state=state,
            measurement=measurement,
            action=action,
            next_state=next_state,
            prediction_samples=prediction_samples,
        )
        reconstruction_loss = (
            terms["state_reconstruction"] + terms["next_state_reconstruction"]
        )
        return (
            reconstruction_weight * reconstruction_loss
            + latent_prediction_weight * terms["latent_prediction"]
            + observation_prediction_weight * terms["observation_prediction"]
            + kl_weight * terms["kl"]
        )

    def kl_loss(self) -> torch.Tensor:
        """Return variational KL for the Bayesian transition model."""

        return self.transition_model.kl_loss()

    def normalize_prediction_target(self, next_state: torch.Tensor) -> torch.Tensor:
        """Normalize VAS targets to match decoded BEV predictions."""

        return self.normalize_birdview(next_state)

    def normalize_birdview(self, birdview: torch.Tensor) -> torch.Tensor:
        """Convert Roach raw masks to float normalized ``[0, 1]`` tensors."""

        if birdview.dim() != 4:
            raise ValueError("birdview must have shape [batch, channels, height, width]")
        if tuple(birdview.shape[1:]) != self.bev_shape:
            raise ValueError(
                "birdview shape mismatch: expected [batch, {}], got {}".format(
                    self.bev_shape,
                    tuple(birdview.shape),
                )
            )
        normalized = birdview.float()
        if normalized.detach().max().item() > 1.0:
            # Roach masks are commonly uint8 images; tests may already pass
            # normalized floats.
            normalized = normalized / 255.0
        return normalized.clamp(0.0, 1.0)

    def _compose_transition_input(
        self,
        latent: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        self._check_latent(latent)
        self._check_2d("action", action, self.action_dim)
        parts = [latent]

        if self.measurement_dim:
            if measurement is None:
                raise ValueError("measurement is required when measurement_dim > 0")
            self._check_2d("measurement", measurement, self.measurement_dim)
            parts.append(measurement)
        elif measurement is not None and measurement.numel() > 0:
            raise ValueError("measurement was provided but measurement_dim is 0")

        if action.shape[0] != latent.shape[0]:
            raise ValueError("latent and action batch sizes differ")
        parts.append(action)
        return torch.cat(parts, dim=1)

    def _check_latent(self, latent: torch.Tensor) -> None:
        self._check_2d("latent", latent, self.latent_dim)

    @staticmethod
    def _check_2d(name: str, value: torch.Tensor, expected_dim: int) -> None:
        if value.dim() != 2:
            raise ValueError("{} must be a 2D tensor [batch, dim]".format(name))
        if value.shape[1] != expected_dim:
            raise ValueError(
                "{} dim mismatch: expected {}, got {}".format(
                    name, expected_dim, value.shape[1]
                )
            )


class BayesianDynamicsModel(nn.Module):
    """Vector-state BNN dynamics model.

    This is the lightweight reproduction core for the paper's BNN environment
    model: it predicts ``next_state`` from ``state, measurement, action`` and
    exposes Monte Carlo samples for shifted VAS scoring.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        measurement_dim: int = 0,
        hidden_dims: Iterable[int] = (128, 128),
        prior_std: float = 1.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if measurement_dim < 0:
            raise ValueError("measurement_dim cannot be negative")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.measurement_dim = measurement_dim
        self.network = BayesianMLP(
            input_dim=state_dim + measurement_dim + action_dim,
            output_dim=state_dim,
            hidden_dims=hidden_dims,
            prior_std=prior_std,
        )

    def forward(
        self,
        state: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        return self.network(
            self._compose_input(state=state, measurement=measurement, action=action),
            sample=sample,
        )

    def predict_next_state_samples(
        self,
        state: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        samples = [
            self.forward(state=state, measurement=measurement, action=action, sample=True)
            for _ in range(num_samples)
        ]
        return torch.stack(samples, dim=0)

    def training_loss(
        self,
        state: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
        next_state: torch.Tensor,
        kl_weight: float = 1.0e-4,
    ) -> torch.Tensor:
        prediction = self.forward(
            state=state,
            measurement=measurement,
            action=action,
            sample=True,
        )
        return F.mse_loss(prediction, next_state) + kl_weight * self.kl_loss()

    def kl_loss(self) -> torch.Tensor:
        return self.network.kl_loss()

    def _compose_input(
        self,
        state: torch.Tensor,
        measurement: Optional[torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        self._check_2d("state", state, self.state_dim)
        self._check_2d("action", action, self.action_dim)
        parts = [state]

        if self.measurement_dim:
            if measurement is None:
                raise ValueError("measurement is required when measurement_dim > 0")
            self._check_2d("measurement", measurement, self.measurement_dim)
            parts.append(measurement)
        elif measurement is not None and measurement.numel() > 0:
            raise ValueError("measurement was provided but measurement_dim is 0")

        parts.append(action)
        return torch.cat(parts, dim=1)

    @staticmethod
    def _check_2d(name: str, value: torch.Tensor, expected_dim: int) -> None:
        if value.dim() != 2:
            raise ValueError("{} must be a 2D tensor [batch, dim]".format(name))
        if value.shape[1] != expected_dim:
            raise ValueError(
                "{} dim mismatch: expected {}, got {}".format(
                    name, expected_dim, value.shape[1]
                )
            )


def _validate_bev_shape(bev_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if len(bev_shape) != 3:
        raise ValueError("bev_shape must be (channels, height, width)")
    channels, height, width = [int(value) for value in bev_shape]
    if channels <= 0:
        raise ValueError("BEV channels must be positive")
    if height <= 0 or width <= 0:
        raise ValueError("BEV height and width must be positive")
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError("BEV height and width must be divisible by 32")
    return channels, height, width
