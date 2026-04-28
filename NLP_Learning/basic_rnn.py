import torch.nn as nn
import torch

rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=1)
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

output, h1 = rnn(x, h0)
print(output)
print(h1)

# 一个token一个token送
# x.shape[0]
# x.size(0)
for idx in range(x.size(0)):
    print(x[idx])     # 取得的是三维列表中的一个二维列表
    x0 = x[idx].unsqueeze(dim=0)  # 但是rnn必须接受三维张量
                                  # rnn还有一个输入是h0
    output, h0 = rnn(x0, h0)

# 两者的结果是一模一样的，验证了RNN的循环机制是串行的