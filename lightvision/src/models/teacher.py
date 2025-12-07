import torch.nn as nn
from torchvision import models

class ResNet18Teacher(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18Teacher, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)