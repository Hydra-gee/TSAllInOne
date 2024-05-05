from argparse import Namespace

import torch
import torch.nn as nn
from torch import Tensor

from .units import Encoder, Generator
from .tools import decomposition, segmentation


class Model(nn.Module):
    def __init__(self, expConfig,args: Namespace) -> None:
        super().__init__()
        self.patch_len = expConfig['patch_len']
        self.patch_num = args['patch_num']
        self.season_net = Net(expConfig,args, 'season')
        self.trend_net = Net(expConfig,args, 'trend')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.mean(x, dim=1, keepdim=True)
        x = x - bias
        res, avg = decomposition(x, self.patch_len // 2)
        stride = (x.shape[1] - self.patch_len) // (self.patch_num - 1)
        res = segmentation(res, self.patch_len, stride)
        avg = segmentation(avg, self.patch_len, stride)
        res = self.season_net(res)
        avg = self.trend_net(avg)
        return res + avg + bias


class Net(nn.Module):
    def __init__(self, expConfig,args: Namespace, mode: str) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(args['patch_num'], args['patch_num']), nn.Dropout(args['dropout']))
        self.encoder = Encoder(expConfig,args, mode)  # mode: season or trend or else
        self.generator = Generator(expConfig,args)

    def forward(self, x: Tensor) -> Tensor:
        # batch * dim * patch_num * patch_len
        x = self.input_layer(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.encoder(x)
        return self.generator(x)
