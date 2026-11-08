import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from defect_inspector.training import validate


class ToyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, inputs):
        assert not self.training, "Validation must use eval mode"
        assert not torch.is_grad_enabled(), "Validation must disable gradients"
        self.calls += 1
        return inputs * self.scale


def test_validate_mode_metrics_and_partial_batch():
    model = ToyClassifier()
    model.train()

    # Two correct and two incorrect predictions.
    inputs = torch.tensor([
        [2.0, 0.0],
        [0.0, 2.0],
        [0.0, 2.0],
        [2.0, 0.0],
    ])
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(
        TensorDataset(inputs, labels),
        batch_size=3,
        shuffle=False,
    )

    metrics, actual_labels, scores = validate(
        model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu")
    )

    low = 1.0 / (1.0 + math.exp(2.0))
    high = 1.0 - low
    assert actual_labels == [0, 1, 0, 1]
    assert scores == pytest.approx([low, high, high, low], abs=1e-7)
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}

    for key in (
        "accuracy", "defect_precision", "defect_recall",
        "defect_f1", "average_precision",
    ):
        assert metrics[key] == pytest.approx(0.5)

    # Correct CE = log(1 + exp(-2)); incorrect CE = correct CE + 2.
    expected_loss = math.log1p(math.exp(-2.0)) + 1.0
    assert metrics["loss"] == pytest.approx(expected_loss, abs=1e-6)
    assert metrics["threshold"] == 0.5
    assert model.calls == 2
    assert model.scale.grad is None