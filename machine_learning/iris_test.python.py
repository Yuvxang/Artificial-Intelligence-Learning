from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

## 数据可视化
iris = load_iris()
print(type(iris))
df1 = pd.DataFrame(iris.data, columns=iris.feature_names)
df1['target'] = iris.target
sns.lmplot(data=df1, x='sepal length (cm)', y='sepal width (cm)', hue='target', fit_reg=True)
# fit_reg用来显示拟合线
plt.title("Iris Dataset")
plt.tight_layout() # 调整边界与图匹配
plt.show()

## 数据集划分
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)

## 特征预处理
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
estimator = KNeighborsClassifier()
estimator = GridSearchCV(estimator, {"n_neighbors":[1, 2, 3, 4, 5]}, cv=5)
estimator.fit(x_train, y_train)

print(estimator.best_score_)
print(estimator.best_params_)
print(estimator.best_estimator_)
print(estimator.cv_results_)


y_pred = estimator.predict(x_test)
print(accuracy_score(y_test, y_pred))







