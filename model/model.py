import torch
import torch.nn as nn
from argparse import Namespace

import model.units as unit
from model.tools import decomposition, segmentation, Transpose


class Model(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.patch_len = args.patch_len
        self.patch_num = args.patch_num
        self.season_net = Net(args, 'season')
        self.trend_net = Net(args, 'trend')

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
    def __init__(self, args: Namespace, mode: str) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            Transpose(-1, -2), nn.Linear(args.patch_num, args.patch_num), nn.Dropout(args.dropout), Transpose(-1, -2))
        self.coder = unit.Coder(args, mode)  # mode: season or trend
        self.generator = unit.Generator(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch * dim * num * len
        x = self.input_layer(x)
        x = self.coder(x)
        x = self.generator(x)
        return x
