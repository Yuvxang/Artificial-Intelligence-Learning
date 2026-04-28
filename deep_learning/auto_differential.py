## 自动微分模块计算更优参数
import torch


## pytorch只支持标量对向量的求导。
w0 = torch.tensor(10, requires_grad=True, dtype=torch.float32)
loss = 2 * w0 ** 2    # 自定义损失函数

loss.sum().backward() # 进行求和转标量，然后反向传播
lr = 0.01

w1 = w0.data - lr * w0.grad
print(w1)

if w0.grad is not None:
    w0.grad.zero_()
# 多次需要的梯度清零

## 也可以通过循环进行多次代入损失函数，梯度清零后代入公式。
## 链条：数据转化为张量，张量组成数据集，输入到dataloader数据加载器。

