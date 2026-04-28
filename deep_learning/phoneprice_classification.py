import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def create_dataset():
    # 文件 - dataframe - 张量 - dataset - dataloader

    # 使用pandas 获取x以及y值
    data = pd.read_csv('../datasets/手机价格预测.csv', encoding='utf-8')
    data.info()
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 使用sklearn库 进行数据预处理 如数据集分割以及标准化
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 40)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 将df封装为张量
    xtrain_t = torch.tensor(x_train, dtype=torch.float32)
    xtest_t = torch.tensor(x_test, dtype=torch.float32)
    ytrain_t = torch.tensor(y_train.values, dtype=torch.int64)
    ytest_t = torch.tensor(y_test.values, dtype=torch.int64)

    # 将x与y拼接成dataset
    train_dataset = TensorDataset(xtrain_t, ytrain_t)
    test_dataset = TensorDataset(xtest_t, ytest_t)

    # 获取特征个数以及目标值总类
    feature_num = x.shape[1]
    label_num = len(np.unique(y))

    # 返回需要的所有数据
    return train_dataset, test_dataset, feature_num, label_num
class PhoneClassification(nn.Module):
    def __init__(self, feature_num, label_num):
        # 父类初始化
        super().__init__()

        # 属性设置（可选）
        self.feature_num = feature_num
        self.label_num = label_num

        # 初始化层结构
        self.linear1 = nn.Linear(feature_num, 512)
        self.linear2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, label_num)

        # 参数初始化 暂无，因为有默认初始化方法。
        # nn.init.xavier_uniform_(linear1.weight)
        # 神经网络的结构做成一个菱形 输入层变大，输出层变小会比较好。

    def forward(self, data):
        data = nn.functional.relu(self.linear1(data))
        data = nn.functional.relu(self.linear2(data))
        output = self.output(data)
        return output
        # 使用交叉熵损失函数 不需要对于输出层使用softmax
def train_model(train_dataset, feature_num, label_num):
    # 初始化dataloader
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # 进行正向传播以及反向传播
    model = PhoneClassification(feature_num, label_num)
    # 损失函数
    lossfunc = nn.CrossEntropyLoss()
    # 优化器 params 告诉梯度下降算法对什么参数进行优化 - w/b。
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, betas=(0.9, 0.99))

    # 设定轮次进行训练
    epochs = 50
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        total_samples = 0

        for x_train,y_train in dataloader:
            y_pred = model(x_train)
            loss = lossfunc(y_pred, y_train)

            total_loss += loss
            total_samples += len(x_train)

            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
        print(f"total_loss: {total_loss}, total_samples: {total_samples}")
    torch.save(model.state_dict(), './model.pkl')


def predict_model(test_dataset, feature_num, label_num):
    # dataloader model pkl 一次循环训练
    dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    model = PhoneClassification(feature_num, label_num)
    model.load_state_dict(torch.load('./model.pkl'))
    correct_count = 0
    model.eval()  # 切换为dropout失效，禁止随机失活

    for x_test,y_test in dataloader:
        y_pred = model(x_test) # 激活函数，未经过softmax处理，不是概率值
        print(torch.softmax(y_pred, dim=-1)) # 手动处理成概率值

        y_pred_id = torch.argmax(y_pred, dim=1)
        # 获取预测概率值最高的索引 - 实际就是预测类别
        correct_count += (y_pred_id == y_test).sum()
    acc_rate = correct_count / len(test_dataset)
    print(f"Accuracy: {acc_rate}")





if __name__ == '__main__':
    train_dataset, test_dataset, feature_num, label_num = create_dataset()
    train_model(train_dataset, feature_num, label_num)
    predict_model(test_dataset, feature_num, label_num)
