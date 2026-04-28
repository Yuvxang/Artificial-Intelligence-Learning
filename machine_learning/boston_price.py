# from sklearn.datasets import load_boston 报错
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, RidgeCV

# 准备数据集
import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]) # 特征
target = raw_df.values[1::2, 2] # 标签

## 数据集切割 先切割再标准化是因为标准化的x只允许二维列表
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=22)

## 数据预处理 特征工程
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train) # 训练集用fit_transform 训练并转换
x_test = transfer.transform(x_test)   # 测试集用transform直接转换，训练相当于直接告诉模型答案

## 模型训练
# estimator = LinearRegression(fit_intercept=True)
estimator = SGDRegressor(loss="squared_error", fit_intercept=True, learning_rate="invscaling", eta0=0.01)
estimator.fit(x_train, y_train)

## 模型预测
y_pred = estimator.predict(x_test)
print(mean_squared_error(y_test, y_pred)) # 不能用accuracy_score 因为是用于分类的。
