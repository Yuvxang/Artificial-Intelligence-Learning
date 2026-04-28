# 张量的拼接操作
import torch

t0 = torch.randint(0, 10, size=(4, 5))
t1 = torch.randint(-1, 3, size=(4, 5))

# cat/concat操作 输入的张量除了指定的拼接维度外，其他维度必须一样
cat1 = torch.cat(tensors=(t0, t1), dim=0)
# 这边tensor元组和列表都行。
print(cat1)
# 以行为单位拼接
# 必须相同的意思就是，行可以不一样（如2行和4行拼6行，但是列必须一样，都是n列）

# stack操作。输入的所有张量形状必须完全一样
# 产生一个新维度，在新维度上进行操作。

