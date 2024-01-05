import torch
import torch.nn as nn

import model.units as unit
from model.tools import decomposition, segmentation

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len
        self.segment_num = args.segment_num
        self.season_model = Net(args, 'season')
        self.trend_model = Net(args, 'trend')

    # 输入历史序列，维度为batch*embed_dim*dim
    def forward(self, x):
        res, avg = decomposition(x, self.pred_len)
        stride = (x.shape[1] - self.pred_len) // (self.segment_num - 1)
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
        self.input_layer = nn.Linear(args.pred_len, args.pred_len)
        layer = unit.Layer(args.device, args.pred_len, args.embed_dim, mode)
        if args.individual:
            self.coder = nn.ModuleList([unit.Coder(layer, args.layer_num) for _ in range(args.channel_dim)])
            self.generator = nn.ModuleList([unit.Generator(args.segment_num) for _ in range(args.channel_dim)])
        else:
            self.coder = unit.Coder(layer, args.layer_num)
            self.generator = unit.Generator(args.segment_num)

    def forward(self, x):
        x = self.input_layer(x)
        # batch * dim * num * len
        if self.individual:
            y = []
            for i in range(x.shape[1]):
                tmp = self.coder[i](x[:, i])
                tmp = self.generator[i](tmp)
                y.append(tmp)
            x = torch.stack(y, dim=1)
        else:
            x = self.coder(x)
            x = self.generator(x)
        return x.transpose(1, 2)


# class PeriodNet(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.l_pred = args.l_pred
#         self.l_step = (args.scale - 1) * self.l_pred // (args.n_segment - 1)  # Intervals between Segments
#         layer = unit.Layer(args.device, self.l_pred, args.n_pattern, stable=True)
#         self.coder = unit.Coder(layer, args.n_layer)  # Temporal Dependencies
#         self.generator = unit.Generator(args.n_segment * args.d_in, args.d_out)
#
#     def forward(self, x):
#         x = slicing(x, self.l_step, self.l_pred)
#         x = self.embed(x.transpose(-1, -2)).transpose(-1, -2)
#         x = self.coder(x)
#         return self.generator(x)
#
#
# class TrendNet(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.l_pred = args.l_pred
#         self.d_out = args.d_out
#         self.l_step = (args.scale - 1) * args.l_pred // (args.n_segment - 1)
#         layer = unit.Layer(args.device, args.l_pred, args.n_pattern, stable=False)
#         self.embed = nn.Linear(args.n_segment * args.d_in, args.n_segment * args.d_in)
#         self.coder = unit.Coder(layer, args.n_layer)
#         self.generator = unit.Generator(args.n_segment * args.d_in, args.d_out)
#
#     def forward(self, x):
#         avg = torch.mean(x, dim=-2, keepdim=True)
#         x = x - avg
#         x = slicing(x, self.l_step, self.l_pred)
#         x = self.embed(x.transpose(-1, -2)).transpose(-1, -2)
#         x = self.coder(x)
#         x = self.generator(x)
#         return x + avg[:, :, -self.d_out:]
