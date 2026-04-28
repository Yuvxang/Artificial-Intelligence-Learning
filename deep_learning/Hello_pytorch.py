## 张量相关基础函数
import torch
import numpy as np

## 使用tensor函数创建 - 根据指定数据创建张量
## 标量创建张量
a2 = torch.tensor(data=999)
# print(a2)

## Numpy ndarray转化为tensor
nda = np.random.randn(2, 5)
t3 = torch.tensor(data=nda) ## 数据类型是torch中的浮点。
# print(t3)

## 通过容器创建张量，感觉
t4 = torch.tensor([11, 22, 33])
t5 = torch.tensor([1.99, 2.88, 3.77])

## 查看张量元素数据类型
print(t5.dtype)

## 使用Tensor函数创建 - 根据形状创建张量
## 5个元素的张量
t6 = torch.Tensor(5)
print(t6)

## 如果还是想传入数据
t7 = torch.Tensor([5])
print(t7.dtype)
# 类型会自动转换为浮点数

## 创建指定类型的张量，这些方法和Tensor类似
t8 = torch.IntTensor([5])   # 截断式整数
t9 = torch.FloatTensor([5])
t10 = torch.DoubleTensor(5) # 写5就是形状，写[5]就是数据

