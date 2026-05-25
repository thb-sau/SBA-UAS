"""Replay buffers for uncertainty-aware stabilization.

SBA-UAS keeps ordinary recent experience in standard buffer ``B`` and moves
low-``u_vas`` familiar transitions into buffer ``D`` when ``B`` saturates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Dict, Iterator, List, Optional


@dataclass(frozen=True)
class Transition:
    """A CARLA/Roach-agnostic transition annotated with shifted VAS."""

    state: Any
    measurement: Any
    action: Any
    reward: float
    next_state: Any
    next_measurement: Any
    done: bool
    u_vas: float
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.u_vas)):
            raise ValueError("u_vas must be a finite familiarity score")


@dataclass(frozen=True)
class BufferAddResult:
    """Observable result of inserting one transition."""

    added_to_standard: Optional[Transition] = None
    migrated_to_familiar: Optional[Transition] = None
    discarded_from_familiar: Optional[Transition] = None


class FamiliarExperienceBuffer:
    """Buffer ``D`` for low-uncertainty familiar historical transitions."""

    def __init__(self, capacity: int, rng: Optional[random.Random] = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._rng = rng or random.Random()
        self._items: List[Transition] = []

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: Transition) -> Optional[Transition]:
        """Add a familiar transition, evicting the largest ``u_vas`` if full."""

        if len(self._items) < self.capacity:
            self._items.append(transition)
            return None

        evict_index = self._choose_extreme_index(minimum=False)
        evicted = self._items[evict_index]
        self._items[evict_index] = transition
        return evicted

    def _choose_extreme_index(self, minimum: bool) -> int:
        target = min if minimum else max
        target_score = target(item.u_vas for item in self._items)
        candidates = [
            index for index, item in enumerate(self._items) if item.u_vas == target_score
        ]
        return self._rng.choice(candidates)

    def state_dict(self) -> Dict[str, Any]:
        """Return a checkpoint-friendly snapshot of buffer ``D``."""

        return {
            "capacity": self.capacity,
            "items": list(self._items),
        }

    @classmethod
    def from_state_dict(
        cls, state_dict: Dict[str, Any], rng: Optional[random.Random] = None
    ) -> "FamiliarExperienceBuffer":
        """Restore buffer ``D`` from :meth:`state_dict` output."""

        buffer = cls(capacity=int(state_dict["capacity"]), rng=rng)
        buffer._items = list(state_dict.get("items", []))
        buffer._validate_restored_size()
        return buffer

    def _validate_restored_size(self) -> None:
        if len(self._items) > self.capacity:
            raise ValueError("restored items exceed buffer capacity")


class StandardReplayBuffer:
    """Buffer ``B`` for recent training transitions."""

    def __init__(
        self,
        capacity: int,
        familiar_buffer: Optional[FamiliarExperienceBuffer] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.familiar_buffer = familiar_buffer
        self._rng = rng or random.Random()
        self._items: List[Transition] = []

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: Transition) -> BufferAddResult:
        """Store a transition in ``B`` and migrate the most familiar old item."""

        if len(self._items) < self.capacity:
            self._items.append(transition)
            return BufferAddResult(added_to_standard=transition)

        migrate_index = self._choose_extreme_index(minimum=True)
        migrated = self._items[migrate_index]
        self._items[migrate_index] = transition

        discarded = None
        if self.familiar_buffer is not None:
            discarded = self.familiar_buffer.add(migrated)

        return BufferAddResult(
            added_to_standard=transition,
            migrated_to_familiar=migrated,
            discarded_from_familiar=discarded,
        )

    def _choose_extreme_index(self, minimum: bool) -> int:
        target = min if minimum else max
        target_score = target(item.u_vas for item in self._items)
        candidates = [
            index for index, item in enumerate(self._items) if item.u_vas == target_score
        ]
        return self._rng.choice(candidates)

    def state_dict(self) -> Dict[str, Any]:
        """Return a checkpoint-friendly snapshot of buffer ``B`` only."""

        return {
            "capacity": self.capacity,
            "items": list(self._items),
        }

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, Any],
        familiar_buffer: Optional[FamiliarExperienceBuffer] = None,
        rng: Optional[random.Random] = None,
    ) -> "StandardReplayBuffer":
        """Restore buffer ``B`` and attach an already-restored buffer ``D``."""

        buffer = cls(
            capacity=int(state_dict["capacity"]),
            familiar_buffer=familiar_buffer,
            rng=rng,
        )
        buffer._items = list(state_dict.get("items", []))
        buffer._validate_restored_size()
        return buffer

    def _validate_restored_size(self) -> None:
        if len(self._items) > self.capacity:
            raise ValueError("restored items exceed buffer capacity")
