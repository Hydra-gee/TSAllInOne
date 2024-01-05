import torch
import torch.nn as nn

import model.unit as unit
from model.tool import slicing, decomposition


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.l_pred = args.l_pred
        self.model_alpha = PeriodNet(args)
        self.model_mu = TrendNet(args)

    # 输入历史序列，维度为batch*embed_dim*dim
    def forward(self, x):
        res, avg = decomposition(x, 2 * self.l_pred)
        res = self.model_alpha(res)
        avg = self.model_mu(avg)
        return res, avg


class PeriodNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.l_pred = args.l_pred
        self.l_step = (args.scale - 1) * self.l_pred // (args.n_segment - 1)  # Intervals between Segments
        self.embed = nn.Linear(args.n_segment * args.d_in, args.n_segment * args.d_in)  # Spatial Dependencies
        layer = unit.Layer(args.device, self.l_pred, args.n_pattern, stable=True)
        self.coder = unit.Coder(layer, args.n_layer)  # Temporal Dependencies
        self.generator = unit.Generator(args.n_segment * args.d_in, args.d_out)

    def forward(self, x):
        x = slicing(x, self.l_step, self.l_pred)
        x = self.embed(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.coder(x)
        return self.generator(x)


class TrendNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.l_pred = args.l_pred
        self.d_out = args.d_out
        self.l_step = (args.scale - 1) * args.l_pred // (args.n_segment - 1)
        layer = unit.Layer(args.device, args.l_pred, args.n_pattern, stable=False)
        self.embed = nn.Linear(args.n_segment * args.d_in, args.n_segment * args.d_in)
        self.coder = unit.Coder(layer, args.n_layer)
        self.generator = unit.Generator(args.n_segment * args.d_in, args.d_out)

    def forward(self, x):
        avg = torch.mean(x, dim=-2, keepdim=True)
        x = x - avg
        x = slicing(x, self.l_step, self.l_pred)
        x = self.embed(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.coder(x)
        x = self.generator(x)
        return x + avg[:, :, -self.d_out:]
