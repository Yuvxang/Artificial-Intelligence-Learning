import pandas as pd
import matplotlib.pyplot as plt
import joblib # 保存模型，无需重复训练
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score
## 手写数字识别.

## 读取csv转成dataframe, 显示这个图片
df1 = pd.read_csv('手写数字识别.csv')
x = df1.iloc[:, 1:] / 255 # 一列之后所有数据
y = df1.iloc[:, 0] # 零列所有行


data_ = x.iloc[3].values
data_ = data_.reshape(28, 28)
plt.axis('off')
plt.imshow(data_, cmap='gray')
plt.show()

## 手写数字识别

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21, stratify=y)
estimator = KNeighborsClassifier(n_neighbors=3)
estimator.fit(x_train, y_train)
score = estimator.score(x_test, y_test)
print(score)

joblib.dump(estimator, 'handwriting.pkl')


