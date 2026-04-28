from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import calinski_harabasz_score
matplotlib.use('TkAgg') # 解决问题

## 准备数据
x, y = make_blobs(
    n_samples=1000 , # 样本数
    n_features=4,  # 特征数
    centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],     # 聚类数
    cluster_std=[0.4, 0.2, 0.2, 0.3]# 聚类标准差
    )
plt.scatter(x[:, 0], x[:, 1])
plt.show()

## 使用模型
cluster = KMeans(n_clusters=4 )
y_pred = cluster.fit_predict(x)

## 颜色划分类别
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()

