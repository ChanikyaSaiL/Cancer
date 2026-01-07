import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x, y):
        q = self.query(x)
        k = self.key(y)
        v = self.value(y)
        attn = torch.softmax((q * k).sum(-1, keepdim=True), dim=1)
        return attn * v
