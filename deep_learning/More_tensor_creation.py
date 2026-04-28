## 使用类numpy中函数的形式创建tensor
import torch

## 随机数种子
torch.manual_seed(129)
t = torch.randint(3, 7, (3, 4))
print(t)

## 创建线性张量
at1 = torch.arange(3, 5, 1)
# 类似于range左闭右开，起点、终点、步长。
print(at1)
# 等差数列
lst = torch.linspace(1, 6, 6)
# 起点、终点、元素个数。


## randint 随机整数
rit = torch.randint(low=1, high=2, size=(4, ))
# 左闭右开区间，起点，终点，元素个数
# 元组只写一个元素需要加逗号
rit2 = torch.randint(low=1, high=2, size=(4, 5, 6))
# 三维 4是4个大数组，5是每个数组5列（5小数组），6是每列6个元素

# rand 随机小数
rit3 = torch.rand(4)
## 生成指定个数小数，形成一个一维列表
rit4 = torch.rand(size=(4, 5))
# randn 标准正态分布 均值为0，方差为1
rit5 = torch.randn((4, 5))
print(rit5)

## 创建01指定值张量
# ones ones_like zeros zeros_like full fill_like
rit6 = torch.ones(size=(4, 5))
## 主要就是指定size，指定矩阵的形状
rit7 = torch.ones_like(rit5)
## 这个是仿照其他的张量的形状进行创建。
# zero类似，不做演示
# full
rit8 = torch.full(size=(4, 5), fill_value=5)
# full_like类似，只需要加入fill_value就可以。
# 这几个用的最多的是zeros

