import numpy as np

from sba_uas.stabilization.replay_buffer import Transition
from sba_uas.training.roach_env_adapter import (
    RoachObservationAdapter,
    RoachTransitionAdapter,
    transitions_to_tensor_batch,
)


class FixedAnnotator:
    def annotate(
        self,
        state,
        measurement,
        action,
        reward,
        next_state,
        next_measurement,
        done,
        metadata=None,
    ):
        return Transition(
            state=state.detach().cpu()[0],
            measurement=measurement.detach().cpu()[0],
            action=action.detach().cpu()[0],
            reward=reward,
            next_state=next_state.detach().cpu()[0],
            next_measurement=next_measurement.detach().cpu()[0],
            done=done,
            u_vas=0.25,
            metadata=metadata,
        )


def roach_obs(offset=0):
    birdview = np.full((2, 15, 32, 32), offset, dtype=np.uint8)
    state = np.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        ],
        dtype=np.float32,
    )
    return {"birdview": birdview, "state": state}


def test_roach_observation_adapter_preserves_birdview_and_measurement_batches():
    batch = RoachObservationAdapter().to_tensor_batch(roach_obs())

    assert tuple(batch.birdview.shape) == (2, 15, 32, 32)
    assert tuple(batch.measurement.shape) == (2, 6)
    assert str(batch.birdview.dtype) == "torch.uint8"


def test_roach_transition_adapter_annotates_vectorized_step():
    adapter = RoachTransitionAdapter()

    transitions = adapter.annotate_step(
        obs=roach_obs(offset=1),
        actions=np.zeros((2, 2), dtype=np.float32),
        rewards=np.array([1.0, 2.0], dtype=np.float32),
        next_obs=roach_obs(offset=2),
        dones=np.array([False, True]),
        infos=[{"town": "Town01"}, {"town": "Town02", "task_idx": 3}],
        annotator=FixedAnnotator(),
    )

    assert len(transitions) == 2
    assert transitions[0].reward == 1.0
    assert transitions[1].done is True
    assert transitions[1].metadata == {"town": "Town02", "task_idx": 3}
    assert transitions[0].u_vas == 0.25


def test_transitions_to_tensor_batch_builds_policy_obs_dicts():
    transitions = RoachTransitionAdapter().annotate_step(
        obs=roach_obs(offset=1),
        actions=np.zeros((2, 2), dtype=np.float32),
        rewards=np.array([1.0, 2.0], dtype=np.float32),
        next_obs=roach_obs(offset=2),
        dones=np.array([False, True]),
        infos=[{}, {}],
        annotator=FixedAnnotator(),
    )

    batch = transitions_to_tensor_batch(transitions)
    policy_obs = batch.to_policy_obs()
    next_policy_obs = batch.next_to_policy_obs()

    assert tuple(batch.state.shape) == (2, 15, 32, 32)
    assert tuple(batch.action.shape) == (2, 2)
    assert tuple(batch.reward.shape) == (2,)
    assert policy_obs["birdview"].shape == (2, 15, 32, 32)
    assert next_policy_obs["state"].shape == (2, 6)
