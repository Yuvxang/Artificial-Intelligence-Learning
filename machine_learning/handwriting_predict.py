import joblib
import matplotlib.pyplot as plt
# 忽略警告
import warnings
warnings.filterwarnings("ignore", module="sklearn")

img = plt.imread('demo.png')
plt.imshow(img, cmap='gray')
x = img.reshape(1, -1)

knn = joblib.load('handwriting.pkl')
y_pred = knn.predict(x)

print(y_pred[0])