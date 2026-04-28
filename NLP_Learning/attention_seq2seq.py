import torch.nn as nn
import torch
import jieba

sentence = "我爱玩原神"
## 添加了注意力机制的Seq2seq
listword = jieba.lcut(sentence, cut_all=True)
print(listword)

class MyAtti(nn.Module):
    def __init__(self, query_size, key_size, value_size, weighted_size, c_size, input_size):
        # 初始化父类
        super().__init__()

        #设置属性值 QKV
        self.query_size = query_size # 张量q最后一维的形状
        self.key_size = key_size     # 张量k最后一位
        self.value_size = value_size #
        self.weighted_size = weighted_size # 相似性和权重张量中，最后一维的形状
        self.c_size = c_size         # 中间语义张量c最后一维
        self.input_size = input_size # 输入到decoder GRU中最后一维的形状

        # 搭建网络结构
        # 第一个用来计算QK相似性
        self.attn_linear = nn.Linear(self.query_size+self.key_size, self.weighted_size)
        self.attn_combine = nn.Linear(self.query_size+self.c_size, self.input_size)

    # 前向传播
    def forward(self, Q, K, V):
        # 1.QK拼接
        QK = torch.concat([Q, K], dim=-1)

        # QK相似性
        t_simu = self.attn_linear(QK)

        # 相似性转权重
        weight = torch.softmax(t_simu, dim=-1)

        # 权重和V矩阵乘法
        C = torch.bmm(weight, V)

        # C与Q拼接
        CQ = torch.concat([C, Q], dim=-1)

        # 形状调整
        input = self.attn_combine(CQ)
        return input, weight

if __name__ == '__main__':
    batch_size = 1
    hidden_layer = 1
    input_size = 5
    hidden_size = 5
    seq_length = 5

    Q = torch.randn(hidden_layer, batch_size, hidden_size)
    K = torch.randn(hidden_layer, batch_size, hidden_size)
    V = torch.randn(batch_size, seq_length, hidden_size)

    query_size = hidden_size
    key_size = hidden_size
    value_size = hidden_size
    weighted_size =seq_length
    c_size = hidden_size
    input_size = hidden_size

    attn_model = MyAtti(query_size, key_size, value_size, weighted_size, c_size, input_size)
    output, weight = attn_model(Q, K, V)
    print(output)
    print(weight)