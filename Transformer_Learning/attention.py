import torch.nn as nn
import torch
import math

def attention(q, k, v, Tensor, mask:torch.Tensor=None, dropout=None):
    # 放入掩码这样就可以兼容三种多头注意力层
    # QKV相同 Mask=None 编码器为多头自注意力
    # KV相同 Mask=None 编码器为多头交叉注意力层
    # QKV相同 Mask不为None 编码器为掩码多头自注意力

    # 实际上是实现softmax(q * kT)/ √dk) * V
    # 根据三个qkv是否相同来确认是否为自注意力。

    # 张量形状
    # batch_size 批次样本个数
    # seq_len 每条句子里有几个词
    # d_model 词向量的维度/隐藏状态维度
    # 这边是为了简化设置为一样的，词向量和隐藏状态维度相同为d_model

    # Q: [batch_size, seq_len, d_model]
    # K: [batch_size, seq_len, d_model]
    # V:[batch_size, seq_len, d_model]
    # mask [batch_size, seq_len, seq_len]

    d_k = q.shape[-1] # size也可以的 d_model
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 核心 bmm有要求，第一维必须相等，两个张量的batch_size不等会直接报错。
    # matmul QK形状不等也可以广播
    # scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # weight = torch.softmax(scores, dim=-1)
    # c = torch.matmul(weight, value)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # mask == 0 如果mask中等于0，score中的对应位置做掩码，原数据更换为-1e9
    weight = torch.softmax(scores, dim=-1)

    if dropout is not None:
        weight = dropout(weight)

    # if dropout is not None:
    #   weight = dropout(weight)
    # dropout是层对象

    C = torch.bmm(weight, v)
    return C, weight

