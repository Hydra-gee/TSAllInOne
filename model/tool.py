import math

import torch


def decomposition(x, l_window):
    # batch * len * channel
    left = x[:, :1, :].repeat(1, math.floor(l_window / 2), 1)
    right = x[:, -1:, :].repeat(1, math.ceil(l_window / 2), 1)
    c = torch.cat([left, x, right], dim=1)
    window = torch.nn.AvgPool1d(l_window + 1, stride=1)  # 滑动窗口的长度就设置为预测长度即可
    avg = window(c.transpose(1, 2)).transpose(1, 2)
    res = x - avg
    return res, avg


def slicing(x, l_step, l_segment):
    batch, l_seq = x.shape[0], x.shape[1]
    assert l_segment % l_step == 0
    assert l_seq % l_segment == 0
    slice_time = l_segment // l_step
    x = x.transpose(1, 2)
    result = []
    for i in range(slice_time):
        seq = x[:, :, i * l_step: l_seq - l_segment + i * l_step].reshape(batch, -1, l_segment)
        result.append(seq)
    result.append(x[:, :, -l_segment:].reshape(batch, -1, l_segment))
    sliced = torch.cat(result, dim=1)
    return sliced
