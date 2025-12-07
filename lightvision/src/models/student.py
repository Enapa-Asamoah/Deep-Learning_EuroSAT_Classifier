import torch.nn as nn
from torchvision import models

class MobileNetV2Student(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2Student, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)