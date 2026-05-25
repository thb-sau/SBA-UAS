import torch

from sba_uas.stabilization.synaptic_importance import (
    SynapticIntelligenceImportance,
)


def test_synaptic_importance_accumulates_positive_gradient_path_contribution():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    tracker = SynapticIntelligenceImportance.from_model(model, damping=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    loss = model.weight.pow(2).sum()
    loss.backward()
    optimizer.step()
    importance = tracker.update(model)

    # delta = -0.2, grad = 2.0, omega = delta * -grad = 0.4
    # omega_bar = abs(delta) * abs(grad) = 0.4
    assert torch.allclose(importance["weight"], torch.tensor([[0.4 / 0.41]]))


def test_synaptic_importance_clips_negative_importance_to_zero():
    model = torch.nn.Linear(1, 1, bias=False)
    tracker = SynapticIntelligenceImportance.from_model(model, damping=0.01)

    with torch.no_grad():
        model.weight.add_(1.0)
    model.weight.grad = torch.ones_like(model.weight)
    importance = tracker.update(model)

    assert torch.allclose(importance["weight"], torch.zeros_like(model.weight))


def test_synaptic_importance_state_restore_preserves_importance():
    model = torch.nn.Linear(1, 1, bias=False)
    tracker = SynapticIntelligenceImportance.from_model(model, damping=0.5)
    with torch.no_grad():
        model.weight.sub_(0.25)
    model.weight.grad = torch.full_like(model.weight, 2.0)
    expected = tracker.update(model)

    restored = SynapticIntelligenceImportance.from_model(model, damping=999.0)
    restored.load_state_dict(tracker.state_dict())

    assert restored.damping == 0.5
    assert torch.allclose(restored.importance()["weight"], expected["weight"])
