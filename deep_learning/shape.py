# 张量形状操作
import torch
# reshape squeeze unsqueeze transpose permute
# 一般使用reshape/unsqueeze

t0 = torch.randint(0, 10, size=(4, 5))
print(f"形状{t0.shape}")
print(f"行:{t0.shape[0]}") # 或者shape改为size也行
print(f"列:{t0.shape[1]}")

# reshape
# 一个是tensor，一个是形状的列表。
t1 = torch.reshape(t0, [5, 4])
t2 = torch.reshape(t0, [2, 5, 2])
# 可以将其中一个作为-1，表示系统自动计算
t3 = torch.reshape(t0, [5, -1, 2])

# squeeze/unsqueeze
# 一个是升维 一个是降维 前后的元素个数必须一致
t4 = torch.squeeze(t0)
t5 = torch.unsqueeze(t0)
# 有一个参数是dim，对于某一个位置的维度降维，如果为<=1，降维成功
# 不成功就不会变
# 如果是unsqueeze就是升维，将对应的维度变为1.

## transpose permute 交换张量的维度
# permute包含了transpose transpose只能交换两个位置，permute可以交换多个
# 元素个数必须一致
t6 = torch.transpose(t0, 1, 2) # / t0.transpose(1, 2),两个参数无顺序
t7 = torch.permute(t0, [1, 2]) # 将变化之后的顺序写进dims里
