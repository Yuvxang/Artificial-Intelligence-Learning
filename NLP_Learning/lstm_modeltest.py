import torch.nn as nn
import torch

model = nn.LSTM(5, 6, 2, bidirectional=False)
biLSTM = torch.nn.LSTM(5, 6, 1, bidirectional=True)
# 三个参数 输入维度，隐藏层维度，隐藏层层数，bidirectional 双向网络结构
# bidirectional开启时，为bi-lstm
# 参数都没变

h0 = torch.zeros(2, 3, 6)
x = torch.randn(3, 3, 5)
c0 = torch.zeros(2, 3, 6)
# c0 上一时间步的细胞状态，和h的形状是一样的，
# 初始的细胞状态/隐藏状态一般使用全零初始化
# 参数都没变

# output, (h0, c0) = model(x, (h0, c0))
output2, (h0, c0) = biLSTM(x, (h0, c0))
# LSTM和RNN的区别主要在细胞状态，h0和c0结合起来作为状态输出
print(output2)
print(h0)
print(c0)
# 分析结果来看，h0和结构的最后一个小矩阵是一样的
# c不需要关注内部的值。
