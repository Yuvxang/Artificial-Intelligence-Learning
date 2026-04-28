import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        # 初始化父类
        super().__init__()

        # 设置属性值（可选）

        # 定义网络结构
        linear1 = nn.Linear(3, 3)
        linear2 = nn.Linear(3, 2)
        output = nn.Linear(2, 2)

        # 参数初始化
        nn.init.xavier_normal_(linear1.weight)
        nn.init.zeros_(linear1.bias)

        nn.init.kaiming_normal_(linear2.weight)
        nn.init.zeros_(linear2.bias)

    def forward(self, data):
        # 线性求和 + 激活函数
        result = nn.functional.sigmoid(self.linear1(data))
        result = nn.functional.relu(self.linear2(result))
        result = nn.functional.softmax(self.output(result), dim=-1)
        return result





