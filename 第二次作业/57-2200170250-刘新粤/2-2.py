import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("D:\\LIUXINYUE\\regress_data1.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
def normalize_data(X):
    X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    return X_norm
X_norm = normalize_data(X)
train_size = int(0.8 * len(X_norm))
X_train_norm, X_test_norm = X_norm[:train_size], X_norm[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
X_train_norm = np.insert(X_train_norm, 0, 1, axis=1)
X_test_norm = np.insert(X_test_norm, 0, 1, axis=1)
def least_squares_fit(X, y):
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta
theta_ls = least_squares_fit(X_train_norm, y_train)
def compute_loss(X, y, theta):
    predictions = X.dot(theta)
    loss = np.sum((predictions - y) ** 2) / (2 * len(y))
    return loss
train_loss_ls = compute_loss(X_train_norm, y_train, theta_ls)
test_loss_ls = compute_loss(X_test_norm, y_test, theta_ls)
print("利用最小二乘法学习到的参数:", theta_ls)
print("训练损失:", train_loss_ls)
print("测试损失:", test_loss_ls)
plt.plot([train_loss_ls], 'ro-', label='Least Squares Train Loss')
plt.plot([test_loss_ls], 'go-', label='Least Squares Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Method')
plt.ylabel('Loss')
plt.legend()
plt.show()