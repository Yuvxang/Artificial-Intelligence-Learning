import torch.nn as nn
import torch

# # 演示batch_size
# model = nn.RNN(5, 6, 1, batch_first=True)
# h0 = torch.randn(1, 3, 6)
# x = torch.randn(3, 3, 5)
# output, h0 = model(x, h0)

# 演示num_layers
model = nn.RNN(5, 6, 2)
h0 = torch.randn(2, 3, 6)
x = torch.randn(3, 3, 5)
output, h0 = model(x, h0)

# 在改变之后，h0无法在output中找到了
# 因为又经过了一次linear

print(output)
print(h0)