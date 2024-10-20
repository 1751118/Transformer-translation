

import numpy as np
import torch
from config import *

def describe(X, description=""):
    print(f"{description}:")
    print(X)
    print(X.shape)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)

    return pad_attn_mask.expand(batch_size, len_q, len_k).to(device) #扩展成 [batch_size, len_q, len_k]适应多头的Q @ KT

def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k = 1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask.to(device)