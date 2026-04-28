import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("../datasets/churn.csv")
data = pd.get_dummies(data, ["Churn", "Gender"])
data.drop(["Churn_No", "Gender_Male"], axis=1, inplace=True)
data.rename(columns={"Churn_Yes": "Flag"}, inplace=True)

## 数据可视化
# def comprehension(x):
#     if x == 0:
#         return "Yes"
#     elif x == 1:
#         return "No"
#
#
# x1 = list(map(comprehension, data["Contract_Month"]))
sns.countplot(data=data, x=data["Contract_Month"], hue="Flag")
plt.show()


x = data[["Contract_Month", "internet_other", "PaymentElectronic"]]
y = data["Flag"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

estimator = LogisticRegression(solver='liblinear', penalty='l2', C=10)
estimator.fit(x_train, y_train)
y_pred = estimator.predict(x_test)
print(classification_report(y_test, y_pred))

