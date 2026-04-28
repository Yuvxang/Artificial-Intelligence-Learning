import torch.nn as nn
import torch

# 掩码张量

def torch_mask():
    t = torch.ones(5, 5)
    print(t)
    # 首先就是创建一个全1矩阵张量


    u_0 = torch.triu(t, diagonal=0) # 上三角掩码
    print(u_0)
    # 然后使用torch.triu函数来将部分数据变为0
    # 上三角
    # diagonal 对角线为几

    u_1 = torch.tril(t, diagonal=0)
    # 下三角
    print(u_1)

# 实际上就用在掩码多头自注意力。

if __name__ == '__main__':
    torch_mask()