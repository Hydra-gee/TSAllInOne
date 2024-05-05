import torch
from torch import Tensor


def decomposition(x: Tensor, kernel_len: int) -> (Tensor, Tensor):
    # batch * len * dim
    if kernel_len % 2 == 0:
        kernel_len += 1
    left = x[:, :1].repeat(1, (kernel_len - 1) // 2, 1)
    right = x[:, -1:].repeat(1, (kernel_len - 1) // 2, 1)
    window = torch.nn.AvgPool1d(kernel_len, stride=1)  # 滑动窗口的长度就设置为预测长度即可
    avg = window(torch.cat([left, x, right], dim=1).transpose(1, 2)).transpose(1, 2)
    res = x - avg
    return res, avg


def segmentation(x: Tensor, kernel_len: int, stride: int) -> Tensor:
    # batch * len * dim
    assert (x.shape[1] - kernel_len) % stride == 0
    x = x.transpose(1, 2)
    return x.unfold(-1, kernel_len, stride)  # batch * dim * patch_num * kernel_len
