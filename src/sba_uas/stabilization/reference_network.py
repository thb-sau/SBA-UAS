"""Reference-network snapshots for parameter stabilization."""

from __future__ import annotations

import copy
from typing import Dict

import torch
from torch import nn


class ReferenceNetwork(nn.Module):
    """Frozen copy of a trainable network used as a historical anchor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.freeze()

    @classmethod
    def from_model(cls, model: nn.Module) -> "ReferenceNetwork":
        """Create a detached frozen deep copy from the current model state."""

        return cls(copy.deepcopy(model))

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.model(*args, **kwargs)

    def freeze(self) -> None:
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
            param.grad = None

    def unfreeze(self) -> None:
        """Allow D-only optimization of the reference copy."""

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad_(True)

    def sync_from(self, model: nn.Module) -> None:
        """Refresh the reference weights from a compatible trainable model."""

        self.model.load_state_dict(model.state_dict())
        self.freeze()

    def parameter_dict(self) -> Dict[str, torch.Tensor]:
        """Return named reference parameters without wrapper prefixes."""

        return {name: param for name, param in self.model.named_parameters()}
