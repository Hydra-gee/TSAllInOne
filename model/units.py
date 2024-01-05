import copy
import torch.nn as nn

from model.attention import Attention

class Layer(nn.Module):
    def __init__(self, device, segment_len, embed_dim, mode, dropout=0):
        super().__init__()
        self.mode = mode
        self.encode = nn.Linear(segment_len, embed_dim)
        self.decode = nn.Linear(embed_dim, segment_len)
        self.attn = Attention(device, segment_len, embed_dim, mode)
        self.func = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        coding = self.encode(x)
        x_new = self.func(self.decode(self.attn(x, x, coding)))
        return self.dropout(x_new) + x


class Coder(nn.Module):
    def __init__(self, layer, layer_num):
        super().__init__()
        self.layer_list = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x


# 最后通过一个线性层对所有序列做叠加
class Generator(nn.Module):
    def __init__(self, segment_num, dropout=0):
        super().__init__()
        self.layer = nn.Linear(segment_num, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # batch * dim * seg_num * seg_len
        return self.dropout(self.layer(x.transpose(-1, -2)).squeeze(-1))  # batch * dim * len
