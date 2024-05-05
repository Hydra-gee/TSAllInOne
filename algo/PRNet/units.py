import copy
from argparse import Namespace

import torch
import torch.nn as nn
from torch import Tensor

from .attention import Attention


class Layer(nn.Module):
    def __init__(self, device: torch.device, patch_len: int, hidden_dim: int, mode: str, dropout: float = 0) -> None:
        super().__init__()
        self.fc_enc = nn.Sequential(nn.Linear(patch_len, hidden_dim), nn.Dropout(dropout))
        self.fc_dec = nn.Sequential(nn.Linear(hidden_dim, patch_len), nn.Dropout(dropout))
        self.attn = Attention(device, mode)
        self.func = nn.LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        x_new = self.fc_enc(x)
        x_new = self.attn(x, x, x_new)
        x_new = self.fc_dec(x_new)
        return self.func(x_new) + x


class Encoder(nn.Module):
    def __init__(self, args: Namespace, mode: str) -> None:
        super().__init__()
        layer = Layer(torch.device('cuda', 0), args['patch_len'], args['hidden_dim'], mode, args['dropout'])
        self.layer_list = nn.ModuleList([copy.deepcopy(layer) for _ in range(args['layer_num'])])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layer_list:
            x = layer(x)
        return x


# 最后通过一个线性层对所有序列做叠加
class Generator(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.proj_layer = nn.Sequential(nn.Linear(args['patch_len'], args['hidden_dim']), nn.Dropout(args['dropout']), nn.LeakyReLU(0.2),
                                        nn.Linear(args['hidden_dim'], args['pred_len']), nn.Dropout(args['dropout']))
        self.out_layer = nn.Sequential(nn.Linear(args['patch_num'], 1), nn.Dropout(args['dropout']))

    def forward(self, x: Tensor) -> Tensor:
        # batch * dim * patch_num * patch_len
        x = self.proj_layer(x)  # batch * dim * patch_num * pred_len
        return self.out_layer(x.transpose(-1, -2)).squeeze(-1).transpose(-1, -2)  # batch * pred_len * dim
