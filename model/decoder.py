import torch
from torch import nn

from model.layers.embedding import PositionalEncoding
from utils import describe
from utils import get_attn_pad_mask, get_attn_subsequence_mask
from model.layers.multi_head_attention import MultiHeadAttention
from model.layers.pos_wise_feed_forward_net import PosWiseFeedForwardNet

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_head)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_head)
        self.pos_wise_fc = PosWiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask) # K、V来自编码器的输出
        dec_outputs = self.pos_wise_fc(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn
        
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, n_layers):
        super().__init__()
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_head, d_ff) for _ in range(n_layers)]
        )

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)

        dec_outputs = self.pos_emb(dec_outputs)
  
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        
        dec_self_attns, dec_enc_attns = [], []

        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns