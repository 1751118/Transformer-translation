

import torch
from torch import nn
import numpy as np
from config import *

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, K, V, mask = None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if mask is not None:
            scores.masked_fill_(mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) #[batch_size, n_heads, len_q, d_v]
        return context, attn 