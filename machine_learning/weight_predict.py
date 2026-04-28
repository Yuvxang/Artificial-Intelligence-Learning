from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 数据导入， 预处理， 特征工程， 模型训练， 模型预测， 模型评估

# 简单数据集
x = [[160], [166], [172], [174], [180]]
y = [56.3, 60.6, 65.1, 68.5, 75.0]
testx = [[176]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)
print(x_train)
print(y_train)

# model.coef_ model_intercept_
# 分别是线性回归的k和b,在训练后可以打印

estimator = LinearRegression()
estimator.fit(x_train, y_train)

print(f"K: {estimator.coef_}")
print(f"b: {estimator.intercept_}")

# 1 y=kx+b
y_pred = estimator.coef_[0] * testx[0][0] + estimator.intercept_
print(round(y_pred, 5))

# 2 预测（实则代入）
y_pred = estimator.predict(testx)
print(round(y_pred[0], 5))
