import copy
import torch.nn as nn

import model.attention as attn


class Layer(nn.Module):
    def __init__(self, device, l_segment, n_pattern, stable=True):
        super().__init__()
        self.state = stable
        self.encode = nn.Linear(l_segment, n_pattern)
        self.decode = nn.Linear(n_pattern, l_segment)
        self.attn = attn.MultiHeadAttention(device, l_segment, n_pattern, stable)
        self.fun = nn.PReLU()
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        patterns = self.encode(x)
        x = self.decode(self.attn(x, x, patterns))
        return self.dropout(self.fun(x)) + x


class Coder(nn.Module):
    def __init__(self, layer, n_layer):
        super().__init__()
        self.n_layer = n_layer
        self.layer_list = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x


# 最后通过一个线性层对所有序列做叠加
class Generator(nn.Module):
    def __init__(self, n_segment, d_out):
        super().__init__()
        self.selector = nn.Linear(n_segment, d_out)
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        # batch*seq_num*embed_dim
        return self.dropout(self.selector(x.transpose(1, 2)))
