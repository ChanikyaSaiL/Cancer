import torch
import torch.nn as nn
from models.histo_encoder import HistoEncoder
from models.photo_encoder import PhotoEncoder
from models.fusion import CrossAttention

class MultimodalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.histo = HistoEncoder()
        self.photo = PhotoEncoder()
        self.attn = CrossAttention()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, histo, photo):
        fh = self.histo(histo)
        fp = self.photo(photo)
        fh_att = self.attn(fh, fp)
        fused = torch.cat([fh_att, fp], dim=1)
        return self.classifier(fused)
