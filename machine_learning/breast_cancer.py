from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score

# 加载数据 - 数据预处理 - 特征工程 - 模型训练 - 模型预测 - 模型评估
# 特征选取，特征预处理，特征降维，特征选择，特征组合

## 加载数据
datas = pd.read_csv("../datasets/breast-cancer-wisconsin.csv")

## 数据的预处理
## 缺失值处理
datas = datas.replace(to_replace="?", value=np.nan)
datas.dropna(inplace=True)

## 特征工程
## 特征与标签的提取
print(datas.head())
x = datas.iloc[:, 1:-1]
y = datas["Class"]

## 数据集分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

## 标准化
transfer = StandardScaler()
transfer.fit_transform(x_train)
transfer.fit(x_test) ## 注意

## 模型训练
estimator = LogisticRegression(solver="liblinear", penalty="l2", C=10)
# estimator = Lasso(alpha=0.1) # 这是l1正则化的api 可能会使得某个特征的权重降为0
#estimator = Ridge(solver="liblinear", alpha=0.5) # 这是l2正则化的api 会使异常特征的权重无限趋近于零
# 正则化是为了使模型不会过拟合，学到太多特征，变得过于复杂。
estimator.fit(x_train, y_train)

## 模型预测
y_pred = estimator.predict(x_test)

## 模型评估
## 只靠准确率不能完全满足各种场景的要求
confusion_matrix = confusion_matrix(y_test, y_pred)
labels = ["正例(异常)", "假例(正常)"]
cm_df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
print(cm_df)
print(f"精确率: {precision_score(y_test, y_pred, pos_label=4)}")
print(f"召回率: {recall_score(y_test, y_pred, pos_label=4)}")
print(f"f1值: {f1_score(y_test, y_pred, pos_label=4)}")


