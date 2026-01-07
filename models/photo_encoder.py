import torch.nn as nn
import torchvision.models as models


class PhotoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(2048, 512)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        return self.fc(x)
