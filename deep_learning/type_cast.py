import torch

data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
data2 = data.type(torch.DoubleTensor)
# type是通用的，可以实现其他类型的转换
data3 = data.type(torch.FloatTensor)
data4 = data.type(torch.IntTensor)
data5 = data.type(torch.LongTensor)

# 或者也可以直接调用对应类型函数
data6 = data.half()
data7 = data.float()
data8 = data.short() # int long double 都行
print(data6.dtype)

# 使用type也可以精细地改变数据类型
data9 = data.type(dtype=torch.float16)
# 其他类似 torch.int64

## Numpy数组 和 张量 之间的转换
# 转换为ndarray
nd = torch.Tensor.numpy(data) # 转换成ndarray 共享内存
ndn = torch.Tensor.numpy(data).copy() # 深拷贝

# 转换为张量 使用torch.tensor即可，不共享内存,或者from_numpy函数。

## 标量张量和数字转换 标量就是只有一个元素的张量
## 使用item
data = torch.tensor([30, ]) # 或者直接data=30
print(data.item())











