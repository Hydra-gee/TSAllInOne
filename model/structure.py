import torch
import torch.nn as nn

import model.units as unit
from model.tools import decomposition, segmentation


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len
        self.seg_num = args.seg_num
        self.season_model = Net(args, 'season')
        self.trend_model = Net(args, 'trend')

    def forward(self, x):
        res, avg = decomposition(x, self.pred_len)
        stride = (x.shape[1] - self.pred_len) // (self.seg_num - 1)
        res = segmentation(res, self.pred_len, stride)
        avg = segmentation(avg, self.pred_len, stride)
        res = self.season_model(res)
        avg = self.trend_model(avg)
        return res, avg


class Net(nn.Module):
    def __init__(self, args, mode):
        super().__init__()
        self.mode = mode
        self.individual = args.individual
        self.pred_len = args.pred_len
        self.input_layer = nn.Sequential(nn.Linear(args.seg_num, args.seg_num), nn.Dropout(args.dropout))
        if args.individual:
            self.coder = nn.ModuleList([unit.Coder(args, mode) for _ in range(args.channel_dim)])
            self.generator = nn.ModuleList([unit.Generator(args.seg_num) for _ in range(args.channel_dim)])
        else:
            self.coder = unit.Coder(args, mode)
            self.generator = unit.Generator(args.seg_num, args.dropout)

    def forward(self, x):
        # batch * dim * num * len
        x = self.input_layer(x.transpose(-1, -2)).transpose(-1, -2)
        if self.individual:
            y = []
            for i in range(x.shape[1]):
                tmp = self.coder[i](x[:, i:i + 1])
                tmp = self.generator[i](tmp)
                y.append(tmp)
            x = torch.cat(y, dim=-1)
        else:
            x = self.coder(x)
            x = self.generator(x)
        return x
