import torch.nn as nn

import model.units as unit
from model.tools import decomposition, segmentation, Transpose


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
        if args.spatial:
            self.spatial_layer = nn.Sequential(Transpose(-1, -3), nn.Linear(args.channel_dim, args.channel_dim), nn.Dropout(args.dropout), Transpose(-1, -3))
        else:
            self.spatial_layer = nn.Sequential()
        self.input_layer = nn.Sequential(Transpose(-1, -2), nn.Linear(args.seg_num, args.seg_num), nn.Dropout(args.dropout), Transpose(-1, -2))
        self.coder = unit.Coder(args, mode)  # mode: season or trend
        self.generator = unit.Generator(args.seg_num, args.dropout)

    def forward(self, x):
        # batch * dim * num * len
        x = self.spatial_layer(x)
        x = self.input_layer(x)
        x = self.coder(x)
        x = self.generator(x)
        return x
