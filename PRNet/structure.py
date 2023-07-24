import torch
import torch.nn as nn

import PRNet.unit as u
from PRNet.settings import args
from PRNet.tool import slicing, decomposition


class Model(nn.Module):
    def __init__(self, l_pred, d_in, d_out, scale):
        super().__init__()
        self.pred_len = l_pred
        self.model_alpha = PeriodNet(l_pred, d_in, d_out, scale)
        self.model_mu = TrendNet(l_pred, d_in, d_out, scale)

    # 输入历史序列，维度为batch*embed_dim*dim
    def forward(self, x):
        res, avg = decomposition(x, 2 * self.pred_len)
        res = self.model_alpha(res)
        avg = self.model_mu(avg)
        return res, avg


class PeriodNet(nn.Module):
    def __init__(self, l_pred, d_in, d_out, scale):
        super().__init__()
        self.pred_len = l_pred
        self.out_dim = d_out
        self.slice_step = (scale - 1) * self.pred_len // (args.slice_num - 1)
        self.embed = nn.Linear(args.slice_num * d_in, args.slice_num * d_in)
        layer = u.Layer(self.pred_len, args.pattern_dim, stable=True)
        self.coder = u.Coder(layer, args.layer_num)
        self.generator = u.Generator(args.slice_num * d_in, d_out)

    def forward(self, x):
        x = slicing(x, self.slice_step, self.pred_len)
        x = self.embed(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.coder(x)
        return self.generator(x)


class TrendNet(nn.Module):
    def __init__(self, l_pred, d_in, d_out, scale):
        super().__init__()
        self.pred_len = l_pred
        self.out_dim = d_out
        self.slice_step = (scale - 1) * l_pred // (args.slice_num - 1)
        layer = u.Layer(l_pred, args.pattern_dim, stable=False)
        self.embed = nn.Linear(args.slice_num * d_in, args.slice_num * d_in)
        self.coder = u.Coder(layer, args.layer_num)
        self.generator = u.Generator(args.slice_num * d_in, d_out)

    def forward(self, x):
        avg = torch.mean(x, dim=-2, keepdim=True)
        x = x - avg
        x = slicing(x, self.slice_step, self.pred_len)
        x = self.embed(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.coder(x)
        x = self.generator(x)
        return x + avg[:, :, -self.out_dim:]
