import math

import torch


def decomposition(x, l_window):
    # batch * len * channel
    left = x[:, :1, :].repeat(1, math.floor(l_window/2), 1)
    right = x[:, -1:, :].repeat(1, math.ceil(l_window/2), 1)
    #print(left.shape, right.shape, x.shape)
    c = torch.cat([left, x, right], dim=1)
    window = torch.nn.AvgPool1d(l_window+1, stride=1)  # 滑动窗口的长度就设置为预测长度即可
    avg = window(c.transpose(1, 2)).transpose(1, 2)
    res = x - avg
    return res, avg


def slicing(x, slice_step, slice_len):
    batch, seq_len = x.shape[0], x.shape[1]
    assert slice_len % slice_step == 0
    assert seq_len % slice_len == 0
    slice_time = slice_len // slice_step
    x = x.transpose(1, 2)
    result = []
    for i in range(slice_time):
        seq = x[:, :, i*slice_step:seq_len-slice_len+i*slice_step].reshape(batch, -1, slice_len)
        result.append(seq)
    result.append(x[:, :, -slice_len:].reshape(batch, -1, slice_len))
    sliced = torch.cat(result, dim=1)
    return sliced
