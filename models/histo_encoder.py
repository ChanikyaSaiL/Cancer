import torch.nn as nn
import torchvision.models as models

class HistoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.densenet121(pretrained=True)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
