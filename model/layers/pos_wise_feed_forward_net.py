
import torch
from torch import nn

class PosWiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)

        return self.layer_norm(residual + output)