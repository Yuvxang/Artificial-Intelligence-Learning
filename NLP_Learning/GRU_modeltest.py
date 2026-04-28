import torch.nn as nn
import torch

gru = nn.GRU(input_size=5, hidden_size=6, num_layers=1)
# 三个参数
# input_size 每一个词向量的维度数
# hidden_size 隐藏层维度数
# num_layer 隐藏层的数量

h0 = torch.randn(1, 3, 6)
# 三个参数
# 第一个是num_layer * num_directions direction是网格方向，一般是1
# 第二个是batch_size
# 第三个是hidden_size 是隐藏层维数

x = torch.randn(1, 3, 5)
# 三个参数
# sequence_length 每一个样本的单词个数，这边每一个样本就是一个词向量
# batch_size
# input_size 输入大小

output, h1 = gru(x, h0)
print(output)
print(h1)
