"""Training and evaluation entry points for SBA-UAS."""

from sba_uas.training.checkpointing import (
    load_checkpoint,
    save_roach_policy_checkpoint,
    save_sba_uas_extra_state,
)
from sba_uas.training.metrics import (
    average_forgetting,
    average_performance,
    forward_transfer,
)
from sba_uas.training.roach_env_adapter import (
    RoachObservationAdapter,
    RoachObservationBatch,
    RoachTransitionAdapter,
    TransitionTensorBatch,
    transitions_to_tensor_batch,
)
from sba_uas.training.trainer import (
    RoachCompatibleSBAUASTrainer,
    SBAUASTrainerConfig,
)


__all__ = [
    "average_forgetting",
    "average_performance",
    "forward_transfer",
    "load_checkpoint",
    "RoachCompatibleSBAUASTrainer",
    "RoachObservationAdapter",
    "RoachObservationBatch",
    "RoachTransitionAdapter",
    "save_roach_policy_checkpoint",
    "save_sba_uas_extra_state",
    "SBAUASTrainerConfig",
    "TransitionTensorBatch",
    "transitions_to_tensor_batch",
]
