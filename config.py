import torch

d_model = 512

d_k = d_v = 64  # K(=Q), V的维度
n_head = 8
n_layers = 6
d_ff = 2048

max_len = 60

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 100