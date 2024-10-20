
import torch
from torch import nn
from model.layers.embedding import PositionalEncoding
from utils import *
from model.layers.multi_head_attention import MultiHeadAttention
from model.layers.pos_wise_feed_forward_net import PosWiseFeedForwardNet
from config import *

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_head)
        self.pos_fc = PosWiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        
        enc_outputs = self.pos_fc(enc_outputs)  # [batch_size, seq, d_model]
        return enc_outputs, attn
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers, d_ff):
        super().__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model).to(device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff) for _ in range(n_layers)])
    
    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, seq_len]
        '''
        enc_outputs = self.src_emb(enc_inputs)                          # [batch_size, seq, d_model]
 
        enc_outputs = self.pos_emb(enc_outputs)                         # [batch_size, seq, d_model]                    
        
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns  # [batch_size, seq, d_model]