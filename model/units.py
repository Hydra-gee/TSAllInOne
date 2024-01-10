import copy
import torch.nn as nn

from model.attention import Attention


class Layer(nn.Module):
    def __init__(self, device, patch_len, mode, dropout=0):
        super().__init__()
        embed_dim = min(patch_len // 2, 48)
        self.encode = nn.Sequential(nn.Linear(patch_len, embed_dim), nn.Dropout(dropout))
        self.decode = nn.Sequential(nn.Linear(embed_dim, patch_len), nn.Dropout(dropout))
        self.attn = Attention(device, mode)
        self.func = nn.LeakyReLU(0.2)

    def forward(self, x):
        x_new = self.encode(x)
        x_new = self.attn(x, x, x_new)
        x_new = self.decode(x_new)
        return self.func(x_new) + x


class Coder(nn.Module):
    def __init__(self, args, mode):
        super().__init__()
        layer = Layer(args.device, args.patch_len, mode, args.dropout)
        self.layer_list = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.layer_num)])

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x


# 最后通过一个线性层对所有序列做叠加
class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj_layer = nn.Sequential(nn.Linear(args.patch_len, args.pred_len), nn.Dropout(args.dropout))
        self.out_layer = nn.Sequential(nn.Linear(args.patch_num, 1), nn.Dropout(args.dropout))

    def forward(self, x):
        # batch * dim * patch_num * patch_len
        x = self.proj_layer(x)
        return self.out_layer(x.transpose(-1, -2)).squeeze(-1).transpose(-1, -2)  # batch * len * dim
