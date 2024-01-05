# 定义对序列的注意力机制
import math

import torch
import torch.nn as nn


def sim_to_weight(sim_matrix):
    # batch * dim * seg_num * seg_num
    origin_shape = sim_matrix.shape
    sim_matrix = sim_matrix.reshape(-1, sim_matrix.shape[-1] * sim_matrix.shape[-2])
    indices = torch.argsort(sim_matrix, dim=-1, descending=True)  # 降序排序，可以保证大的相似度分配小的概率
    prob = (torch.argsort(indices, dim=-1) + 1) / indices.shape[-1]  # 通过位次分配概率
    weights = -torch.log(prob).reshape(origin_shape)  # batch * dim * seg_num * seg_num
    return weights / torch.sum(weights, dim=-1, keepdim=True)


def season_attention(query, key, value):
    # batch * dim * seg_num * seg_len
    assert query.shape[-1] == key.shape[-1]
    score = torch.matmul(query, key.transpose(-1, -2))  # dot product
    q_len = torch.sqrt(torch.sum(query * query, dim=-1, keepdim=True)) + 1e-5  # batch * dim * seg_num * 1
    k_len = torch.sqrt(torch.sum(key * key, dim=-1, keepdim=True)) + 1e-5  # batch * dim * seg_num * 1
    qk_len = torch.matmul(q_len, k_len.transpose(-1, -2))  # batch * dim * seg_num * seg_num
    score = score / qk_len  # cosine similarity
    score = sim_to_weight(score)
    return torch.matmul(score, value)


# 趋势事件attention
def trend_attention(query, key, value):
    # 维度为batch*dim*seq_num*embed_dim
    # 此时query和key均为提取到的序列特征
    assert query.shape[-1] == key.shape[-1]
    query = (query - torch.mean(query, dim=-1, keepdim=True)).unsqueeze(-2)
    key = (key - torch.mean(key, dim=-1, keepdim=True)).unsqueeze(-3)
    score = torch.sum((query - key) * (query - key), dim=-1)
    score = sim_to_weight(score)
    return torch.matmul(score, value)


def norm_attention(query, key, value):
    assert query.shape[-1] == key.shape[-1]
    score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])  # dot product
    score = torch.softmax(score, dim=-1)
    return torch.matmul(score, value)


# 定义多头注意力机制，但可能暂时不考虑使用
class Attention(nn.Module):
    def __init__(self, device, mode):
        super().__init__()
        self.mode = mode
        self.device = device

    def forward(self, query, key, value):
        # batch * dim * seq_num * embed_dim
        if self.mode == 'season':
            value = season_attention(query, key, value)
        elif self.mode == 'trend':
            value = trend_attention(query, key, value)
        else:
            value = norm_attention(query, key, value)
        return value
