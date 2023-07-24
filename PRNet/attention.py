# 定义对序列的注意力机制
import math

import torch
import torch.nn as nn


# 周期事件attention
def corr_attention(query, key, value, device):
    # batch*head_num*seq_num*pattern_num
    assert query.shape[-1] == key.shape[-1]
    score = torch.matmul(query, key.transpose(-1, -2))  # dot product
    len_q = torch.sqrt(torch.sum(query * query, dim=-1)) + 1e-5
    len_k = torch.sqrt(torch.sum(key * key, dim=-1)) + 1e-5
    len_mat = torch.matmul(len_q.unsqueeze(-1), len_k.unsqueeze(-2))
    score = score / len_mat  # cosine similarity
    signal = torch.sign(score)
    size = score.shape
    score = torch.abs(score).reshape(-1)
    _, idx = torch.sort(score, descending=True)
    prob = torch.arange(1, idx.shape[0]+1).to(device)
    prob = prob / idx.shape[0]
    score[idx] = prob
    score = -torch.log(score).reshape(size) * signal
    score = score / torch.sum(torch.abs(score), dim=-1, keepdim=True)
    v_attn = torch.matmul(score, value)
    return v_attn


# 趋势事件attention
def diff_attention(query, key, value, device):
    # 维度为batch*head_num*seq_num*head_dim
    # 此时query和key均为提取到的序列特征
    assert query.shape[-1] == key.shape[-1]
    query = query - torch.mean(query, dim=-1, keepdim=True)
    query = query.unsqueeze(-2)
    key = key - torch.mean(key, dim=-1, keepdim=True)
    key = key.unsqueeze(-3)
    score = torch.mean((query - key) * (query - key), dim=-1)
    size = score.shape
    score = score.reshape(-1)
    _, idx = torch.sort(score, descending=False)
    prob = torch.arange(1, idx.shape[0]+1).to(device)
    prob = prob / idx.shape[0]
    score[idx] = prob
    score = -torch.log(score).reshape(size)
    score = score / torch.sum(score, dim=-1, keepdim=True)
    v_attn = torch.matmul(score, value)
    return v_attn


def norm_attention(query, key, value):
    assert query.shape[-1] == key.shape[-1]
    score = torch.matmul(query, key.transpose(-1, -2))  # dot product
    score = score / math.sqrt(query.shape[-1])  # cosine similarity
    score = torch.softmax(score, dim=-1)
    v_attn = torch.matmul(score, value)
    return v_attn


# 定义多头注意力机制，但可能暂时不考虑使用
class MultiHeadAttention(nn.Module):
    def __init__(self, device, l_seq, n_pattern, state, n_head=1):
        super().__init__()
        assert l_seq % n_head == 0
        assert n_pattern % n_head == 0
        self.state = state
        self.n_head = n_head
        self.device = device

    def forward(self, query, key, value):
        # batch*seq_num*embed_dim
        batch_size, n_segment = query.shape[0], query.shape[1]
        query, key, value = [x.reshape(batch_size, n_segment, self.n_head, -1).transpose(1, 2) for x in (query, key, value)]
        if self.state:
            v_out = corr_attention(query, key, value, self.device)
        else:
            v_out = diff_attention(query, key, value, self.device)
        return v_out.transpose(1, 2).reshape(batch_size, n_segment, -1)
