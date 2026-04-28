# 张量的数值计算
import torch
torch.manual_seed(129)

## 加减乘除
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])
t3 = t1 + t2
t4 = t1 * 10
t5 = t1 + 10 # 每一个数据都+10
print(t3)
# 可以add、sub、mul、div、neg(取反）函数
# add_ sub_ mul_ div_ neg_需要用变量接收，原始数据会受到影响。
# 符号一样可以用 + - * / *-1
# 其中mul也就是*-点乘，只有相同形状的张量才能进行 - 对应位置相乘
# 估计之后还有@也就是矩阵乘法，a列=b行才行。

## 矩阵乘法 matmul @ a列 = b行 得到的是a行b列的数组
tm = torch.randint(-1, 5, (3, 4))
tn = torch.randint(-1, 5, (4, 6))
# print(tm.matmul(tn))
print(tm)

## 对于张量本身数据的运算
# sum 指定dim=0按行求和，指定dim=1按列求和
print(tm.sum(dim=1))
print(tm.sum(dim=0))

# mean tensor必须是浮点数类型 dim=0/1
tm = tm.type(dtype = torch.float32)
print(tm.mean(dim=1))
print(tm.mean(dim=0))
print(tm.mean())

# min/max
# pow
print(torch.pow(tn, 2)) # 平方

# sqrt
# exp
# log2、log10 log就是以e为底
# 了解为之

