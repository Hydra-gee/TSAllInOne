# 定义对序列的注意力机制
import math

import torch
import torch.nn as nn


# 周期事件attention
def corr_attention(query, key, value, device):
    # batch * dim * seg_num * seg_len
    assert query.shape[-1] == key.shape[-1]
    score = torch.matmul(query, key.transpose(-1, -2))  # dot product
    q_len = torch.sqrt(torch.sum(query * query, dim=-1, keepdim=True)) + 1e-5  # batch * dim * seg_num * 1
    k_len = torch.sqrt(torch.sum(key * key, dim=-1, keepdim=True)) + 1e-5  # batch * dim * seg_num * 1
    qk_len = torch.matmul(q_len, k_len.transpose(-1, -2))  #  # batch * dim * seg_num * seg_num
    score = score / qk_len  # cosine similarity
    signal = torch.sign(score)
    size = score.shape
    score = torch.abs(score).reshape(-1)  # 拉直
    _, idx = torch.sort(score, descending=False)  # 获取降序排序的序号
    prob = torch.linspace(0, 1, idx.shape[0], device=device) + 1 / idx.shape[0]  # 生成概率
    score[idx] = prob  # 依据概率为每个向量赋值
    score = -torch.log(score).reshape(size) * signal  # 对概率取log得到信息量
    score = score / torch.sum(torch.abs(score), dim=-1, keepdim=True)  # 归一化
    return torch.matmul(score, value)

def _corr_attention(query, key, value, device):
    # batch * dim * seg_num * seg_len
    assert query.shape[-1] == key.shape[-1]
    score = torch.matmul(query, key.transpose(-1, -2))  # dot product
    q_len = torch.sqrt(torch.sum(query * query, dim=-1, keepdim=True)) + 1e-5  # batch * dim * seg_num * 1
    k_len = torch.sqrt(torch.sum(key * key, dim=-1, keepdim=True)) + 1e-5  # batch * dim * seg_num * 1
    qk_len = torch.matmul(q_len, k_len.transpose(-1, -2))  #  # batch * dim * seg_num * seg_num
    score = score / qk_len  # cosine similarity
    signal = torch.sign(score)
    size = score.shape
    score = torch.abs(score).reshape(-1, size[-2] * size[-1])  # 拉直   batches * num2
    _, idx = torch.sort(score, dim=-1, descending=False)  # 获取降序排序的序号
    prob = torch.linspace(0, 1, idx.shape[-1], device=device).unsqueeze(0).repeat(idx.shape[0], 1) + 1 / idx.shape[0]  # 生成概率
    print(score.shape, idx.shape, prob.shape)
    score[idx] = prob  # 依据概率为每个向量赋值
    score = -torch.log(score).reshape(size) * signal  # 对概率取log得到信息量
    score = score / torch.sum(torch.abs(score), dim=-1, keepdim=True)  # 归一化
    return torch.matmul(score, value)


# 趋势事件attention
def diff_attention(query, key, value, device):
    # 维度为batch*dim*seq_num*embed_dim
    # 此时query和key均为提取到的序列特征
    assert query.shape[-1] == key.shape[-1]
    query = query - torch.mean(query, dim=-1, keepdim=True)
    query = query.unsqueeze(-2)
    key = key - torch.mean(key, dim=-1, keepdim=True)
    key = key.unsqueeze(-3)
    score = torch.mean((query - key) * (query - key), dim=-1)
    size = score.shape
    score = score.reshape(-1)
    _, idx = torch.sort(score, descending=True)
    prob = torch.linspace(0, 1, idx.shape[0], device=device) + 1 / idx.shape[0]  # 生成概率
    score[idx] = prob
    score = -torch.log(score).reshape(size)
    score = score / torch.sum(score, dim=-1, keepdim=True)
    return torch.matmul(score, value)



def norm_attention(query, key, value):
    assert query.shape[-1] == key.shape[-1]
    score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])  # dot product
    score = torch.softmax(score, dim=-1)
    return torch.matmul(score, value)


# 定义多头注意力机制，但可能暂时不考虑使用
class Attention(nn.Module):
    def __init__(self, device, l_seq, n_pattern, mode):
        super().__init__()
        self.mode = mode
        self.device = device

    def forward(self, query, key, value):
        # batch * dim * seq_num * embed_dim
        batch_size, segment_num = query.shape[0], query.shape[2]
        if self.mode == 'season':
            value = corr_attention(query, key, value, self.device)
        elif self.mode == 'trend':
            value = diff_attention(query, key, value, self.device)
        else:
            value = norm_attention(query, key, value)
        return value
