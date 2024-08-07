import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
X_test = np.array([[2, 2], [4, 4], [6, 6]])

y_train = np.array([3, 5, 7, 9])
y_test = np.array([4, 6, 8])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alphas = np.logspace(-5, 5, num=100)

train_losses = []
test_losses = []

for alpha in alphas:
# 使用Ridge模型，设置正则化系数为alpha
model = Ridge(alpha=alpha)

# 拟合模型
model.fit(X_train_scaled, y_train)

# 计算训练误差和测试误差
train_loss = ((model.predict(X_train_scaled) - y_train) ** 2).mean()
test_loss = ((model.predict(X_test_scaled) - y_test) ** 2).mean()

# 存储误差
train_losses.append(train_loss)
test_losses.append(test_loss)

plt.plot(alphas, train_losses, label='Train Loss')
plt.plot(alphas, test_losses, label='Test Loss')
plt.xlabel('Alpha')
plt.ylabel('Loss')
plt.xscale('log')
plt.title('Training and Testing Loss')
plt.legend()
plt.show()