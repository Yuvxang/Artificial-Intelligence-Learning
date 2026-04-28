from sklearn.neighbors import KNeighborsClassifier

x_test = [[1], [2], [3], [4]]
y_test = [0, 0, 1, 1]

estimator = KNeighborsClassifier(n_neighbors=3)
estimator.fit(x_test, y_test)

result = estimator.predict([[4]])
print(result[0])