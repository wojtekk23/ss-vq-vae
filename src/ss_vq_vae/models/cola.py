import torch
from torch import nn
from ..nn import EfficientNetEmbedding, EfficientNetType


class COLA(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(COLA, self).__init__()
        self.encoder = EfficientNetEmbedding(EfficientNetType.EFFICIENTNET_B0, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        x = self.encoder(x)
        x = self.linear(x)

        return x
