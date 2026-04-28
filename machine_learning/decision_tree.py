## 此为CART决策树的演示。
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv("../datasets/train.csv")
datas.info()
y = datas["Survived"]
x = datas[["Age", "Sex", "Pclass"]]

x = x.copy()
x = x.fillna(x["Age"].mean())
x = pd.get_dummies(x, ["Age"])
x.info()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 23)

tree = DecisionTreeClassifier(criterion='gini', max_depth=10)
# criterion 特征选择标准 gini或者entropy entropy就是信息增益ID3。
# 默认是CART，也就是gini.
# min_samples_split 内部节点再划分所需最小样本数 大于这个数节点才能分叉
# min_samples_leaf 叶子节点最少样本数 叶子节点必须有这些个样本，或者剩余的不足的。
# max_depth 最大高度
tree.fit(x_train, y_train)

y_pred = tree.predict(x_test)
print(classification_report(y_pred, y_test))

plt.figure(figsize=(150, 100))
plot_tree(tree, max_depth=10, filled=True)
plt.savefig("decision_tree.png")
plt.show()

