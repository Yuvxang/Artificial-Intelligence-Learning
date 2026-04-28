## 此为随机森林预测的演示。
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd

## 加载数据
data = pd.read_csv("../datasets/train.csv")
data.info()

## 数据的预处理
x = data[["Pclass", "Sex", "Age"]]
y = data["Survived"]

# 缺失值处理
x = x.copy()
x["Age"] = x["Age"].fillna(x["Age"].mean())

# onehot处理
x = pd.get_dummies(x, ["Sex"])

# 分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

## 特征工程 略

## 模型训练
# 单一
estimator = DecisionTreeClassifier()
estimator.fit(x_train, y_train)
y_pred = estimator.predict(x_test)
print(f"准确率：{estimator.score(x_test, y_test)}")

# 随机森林 - 默认参数
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_pred2 = rfc.predict(x_test)
print(f"准确率：{rfc.score(x_test, y_test)}")

# 随机森林 - 网格搜索交叉验证
rfc2 = RandomForestClassifier()
rfc2.fit(x_train, y_train)
param_grid = {
    'n_estimators': [30, 50, 60, 90, 110],
    'max_depth': [2, 3, 5, 7],
}
gs_estimator = GridSearchCV(rfc2, param_grid=param_grid, cv=2)
gs_estimator.fit(x_train, y_train)
print(f"最佳参数：{gs_estimator.best_params_}")
y_pred3 = gs_estimator.predict(x_test)
print(f"准确率：{gs_estimator.score(x_test, y_test)}")
