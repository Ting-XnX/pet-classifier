from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

def build_model(pretrained: bool = True):
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 37)
    return model
