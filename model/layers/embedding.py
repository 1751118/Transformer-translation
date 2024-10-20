
import numpy as np
import torch.nn as nn
import torch
from config import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2*i / d_model) for i in range(d_model)]
            for pos in range(max_len)
        ])

        pos_table[:, ::2] = np.sin(pos_table[:,::2])
        pos_table[:, 1::2] = np.cos(pos_table[:, 1::2])
        self.pos_table = torch.FloatTensor(pos_table).to(device)

    def forward(self, enc_inputs):
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)