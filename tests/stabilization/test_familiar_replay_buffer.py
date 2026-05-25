from sba_uas.stabilization.replay_buffer import (
    FamiliarExperienceBuffer,
    StandardReplayBuffer,
    Transition,
)


def transition(label, u_vas):
    return Transition(
        state=label,
        measurement=None,
        action="action-{}".format(label),
        reward=1.0,
        next_state="next-{}".format(label),
        next_measurement=None,
        done=False,
        u_vas=u_vas,
    )


class PickLastRng:
    def __init__(self):
        self.choices = []

    def choice(self, candidates):
        self.choices.append(list(candidates))
        return candidates[-1]


def test_full_standard_buffer_migrates_most_familiar_transition_to_familiar_buffer():
    familiar = FamiliarExperienceBuffer(capacity=4)
    standard = StandardReplayBuffer(capacity=3, familiar_buffer=familiar)

    standard.add(transition("old-familiar", 0.1))
    standard.add(transition("old-middle", 0.4))
    standard.add(transition("old-novel", 0.9))

    result = standard.add(transition("new", 0.7))

    assert result.migrated_to_familiar.state == "old-familiar"
    assert {item.state for item in standard} == {"old-middle", "old-novel", "new"}
    assert [item.state for item in familiar] == ["old-familiar"]


def test_full_familiar_buffer_evicts_least_familiar_transition():
    familiar = FamiliarExperienceBuffer(capacity=2)
    familiar.add(transition("very-familiar", 0.05))
    familiar.add(transition("least-familiar", 0.8))

    evicted = familiar.add(transition("migrated-familiar", 0.2))

    assert evicted.state == "least-familiar"
    assert {item.state for item in familiar} == {"very-familiar", "migrated-familiar"}


def test_buffers_restore_from_checkpoint_state_without_mixing_standard_and_familiar():
    familiar = FamiliarExperienceBuffer(capacity=2)
    standard = StandardReplayBuffer(capacity=2, familiar_buffer=familiar)
    standard.add(transition("standard-a", 0.3))
    standard.add(transition("standard-b", 0.6))
    familiar.add(transition("familiar-a", 0.1))

    restored_familiar = FamiliarExperienceBuffer.from_state_dict(familiar.state_dict())
    restored_standard = StandardReplayBuffer.from_state_dict(
        standard.state_dict(),
        familiar_buffer=restored_familiar,
    )

    assert restored_standard.capacity == 2
    assert restored_familiar.capacity == 2
    assert [item.state for item in restored_standard] == ["standard-a", "standard-b"]
    assert [item.state for item in restored_familiar] == ["familiar-a"]


def test_equal_u_vas_extremes_are_resolved_with_injected_random_choice():
    rng = PickLastRng()
    familiar = FamiliarExperienceBuffer(capacity=2)
    standard = StandardReplayBuffer(capacity=3, familiar_buffer=familiar, rng=rng)
    standard.add(transition("tie-a", 0.1))
    standard.add(transition("tie-b", 0.1))
    standard.add(transition("novel", 0.8))

    result = standard.add(transition("new", 0.5))

    assert rng.choices == [[0, 1]]
    assert result.migrated_to_familiar.state == "tie-b"
