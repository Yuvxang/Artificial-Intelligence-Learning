import copy
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
    # 相似性得分

    # 核心 bmm有要求，第一维必须相等，两个张量的batch_size不等会直接报错。
    # matmul QK形状不等也可以广播
    # scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # weight = torch.softmax(scores, dim=-1)
    # c = torch.matmul(weight, value)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # mask == 0 如果mask中等于0，score中的对应位置做掩码，原数据更换为-1e9
    # mask可能和scores形状不同。内部有广播机制会变成一样的。
    weight = torch.softmax(scores, dim=-1)

    if dropout is not None:
        weight = dropout(weight)

    # if dropout is not None:
    #   weight = dropout(weight)
    # dropout是层对象

    C = torch.bmm(weight, v)
    return C, weight

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 多头注意力 多头就是对于QKV使用多个线性层 点积
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        # 确保d_model可以被head整除，不整除报错
        assert d_model % n_head == 0

        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        # 搭建网络结构 四个线性层。特点在于输入与输出的维度相同
        """
            1 - 第一个线性层，专门对于Q进行线性转换
            2 - 第二个，专门对K
            3 - 专门对V
            4 - 对于各个头的结果进行线性转换，使得数据满足高斯分布（正态分布）
                （标准化）实际为了让模型训练稳定，缓解梯度消失/梯度爆炸
        """
        self.Linear_List = clones(nn.Linear(d_model, d_model), 4)

        # 存储权重矩阵信息
        self.weighted = None

    def forward(self, q, k, v, mask=None, dropout=None):
        # query/key/value都一样，mask不一样
        # mask:掩码，形状为[head, seq_len, seq_len] 第一个是头数，每一个头的掩码可以不一样

        # 1-掩码处理，需要进行张量升维(多一维1为了进行广播机制）
        # [head, seq_len, seq_len] - [1, head, seq_len, seq_len]
        if mask is not None:
            mask = mask.unsqueeze(dim=0)

        # 2 - 获得句子条数。
        batch_size = q.shape[0]

        # 3 - 前三个线性层分别对QKV进行处理
        # 分开写
        # 将左边列表与右边列表对应。对应不到的不操作。
        self.model_and_data_list = list(zip(self.Linear_List, (q, k, v)))
        self.linear_output_list = []
        for model, data in self.model_and_data_list:
            # 里面是3个元组(linear_1, query)...
            # 线性 - 分头并调整形状 -
            model_output = model(data)
            model_output.reshape(batch_size, -1, self.n_head, self.head_dim)
            model_output = model_output.transpose(1, 2)
            self.linear_output_list.append(model_output)

        # 4 - 计算注意力
        new_query, new_key, new_value = self.linear_output_list
        C, weighted = attention(new_query, new_key, new_value, mask=mask, dropout=dropout)

        # 5 - 线性变换
        result = C.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        return self.Linear_List[-1](result)

