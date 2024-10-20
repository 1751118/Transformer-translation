
import torch
from torch import nn
from config import *
from model.layers.scale_dot_product_attention import ScaleDotProductAttention
from utils import describe

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, n_head):
#         super().__init__()
#         self.W_Q = nn.Linear(d_model, d_k * n_head, bias=False)
#         self.W_K = nn.Linear(d_model, d_k * n_head, bias=False)
#         self.W_V = nn.Linear(d_model, d_v * n_head, bias=False)
#         self.fc = nn.Linear(d_v * n_head, d_model, bias=False)
#         self.layer_norm = nn.LayerNorm(d_model)
#         self.n_head = n_head

#     def forward(self, input_Q, input_K, input_V, mask):
#         '''
#         input_Q, input_K, input_V: [batch_size, seq_len, d_model]
        
#         '''
#         residual, batch_size = input_Q, input_Q.size(0)
#         Q = self.W_Q(input_Q).view(batch_size, -1, self.n_head, d_k).transpose(1, 2)
#         K = self.W_K(input_K).view(batch_size, -1, self.n_head, d_k).transpose(1, 2)        #[batch_size, n_head, seq_len, d_k]
#         V = self.W_V(input_V).view(batch_size, -1, self.n_head, d_v).transpose(1, 2)        #[batch_size, n_head, seq_len, d_v]

#         mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
#         context, attn = ScaleDotProductAttention()(Q, K, V, mask)
#         context = context.transpose(1, 2).reshape(batch_size, -1, self.n_head * d_v)        #[batch_size, seq_len, d_model]

#         output = self.fc(context).to(device)
#         return self.layer_norm(output + residual), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_head, bias=False)
        self.fc = nn.Linear(d_v * n_head, d_model, bias=False)
        self.n_head = n_head
        self.layer_norm = nn.LayerNorm(d_model)  # 将LayerNorm定义为类的成员变量

    def forward(self, input_Q, input_K, input_V, mask):
        '''
        input_Q, input_K, input_V: [batch_size, seq_len, d_model]
        '''
        # 获取当前设备
        device = input_Q.device
        
        residual, batch_size = input_Q, input_Q.size(0)

        # 将权重和中间结果转移到相同的设备上
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_head, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_head, d_k).transpose(1, 2)  #[batch_size, n_head, seq_len, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_head, d_v).transpose(1, 2)  #[batch_size, n_head, seq_len, d_v]

        # 确保 mask 在相同的设备上
        mask = mask.to(device).unsqueeze(1).repeat(1, self.n_head, 1, 1)

        # 计算注意力
        context, attn = ScaleDotProductAttention()(Q, K, V, mask)

        # 恢复维度并确保在相同设备上
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_head * d_v)

        # 将输出进行线性变换并应用 LayerNorm
        output = self.fc(context)

        # 确保 LayerNorm 的操作在相同的设备上
        output = self.layer_norm(output + residual)
        
        return output, attn
