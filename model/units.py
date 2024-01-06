import copy
import torch.nn as nn

from model.attention import Attention


class Layer(nn.Module):
    def __init__(self, device, seg_len, embed_dim, mode, dropout=0):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(seg_len, embed_dim), nn.Dropout(dropout))
        self.decode = nn.Sequential(nn.Linear(embed_dim, seg_len), nn.Dropout(dropout))
        self.attn = Attention(device, mode)
        self.func = nn.ReLU()

    def forward(self, x):
        x_new = self.encode(x)
        x_new = self.attn(x, x, x_new)
        x_new = self.decode(x_new)
        return self.func(x_new) + x


class Coder(nn.Module):
    def __init__(self, args, mode):
        super().__init__()
        layer = Layer(args.device, args.pred_len, args.embed_dim, mode, args.dropout)
        self.layer_list = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.layer_num)])

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x


# 最后通过一个线性层对所有序列做叠加
class Generator(nn.Module):
    def __init__(self, seg_num, dropout=0):
        super().__init__()
        self.layer = nn.Linear(seg_num, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # batch * dim * seg_num * seg_len
        return self.dropout(self.layer(x.transpose(-1, -2)).squeeze(-1)).transpose(-1, -2)  # batch * len * dim
