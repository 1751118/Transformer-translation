

import sys
from pathlib import Path

# 获取当前文件的父目录（即module目录）
current_dir = Path(__file__).resolve().parent

# 获取上层目录（即project目录）
parent_dir = current_dir.parent

# 将上层目录添加到sys.path中
sys.path.append(str(parent_dir))


import torch
from torch import nn

from config import *
from model.encoder import Encoder
from model.decoder import Decoder
from utils import describe

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_head, n_layers):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_head, n_layers, d_ff)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_head, d_ff, n_layers)
        self.linear = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs) # [batch_size, tgt_len, d_model]
        logits = self.linear(dec_outputs)                                                              # [batch_size, tgt_len, tgt_vocab_size]
        return logits.view(-1, logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns  # [batch_size * tgt_len, tgt_vocab_size]