
"""ResNet18 model definition for binary defect classification."""

from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


NUM_CLASSES = 2


def build_model(pretrained: bool = True) -> nn.Module:
    """Build a ResNet18 with two output logits."""

    weights = (
        ResNet18_Weights.IMAGENET1K_V1
        if pretrained
        else None
    )

    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    for parameter in model.parameters():
        parameter.requires_grad_(True)

    return model
