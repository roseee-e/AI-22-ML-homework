import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams
config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)
path = 'D:/qq/regress_data1.csv'
data = pd.read_csv(path)
X_data = data.iloc[:, :-1].copy()
y_data = data.iloc[:, -1].copy()
X_data.insert(0, 'Ones', 1)
X = X_data.values
y = y_data.values
def computeCost(X, y, W):
    m = len(y)
    error = np.dot(X, W) - y
    cost = np.sum(np.square(error)) / (2 * m)
    return cost
def gradientDescent(X, y, W, alpha):
    m = len(y)
    error = np.dot(X, W) - y
    gradient = np.dot(X.T, error) / m
    W = W - alpha * gradient
    return W
def linearRegression(X, y, alpha, iterations):
    cost_history = []
    W = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        cost = computeCost(X, y, W)
        cost_history.append(cost)
        W = gradientDescent(X, y, W, alpha)
    return cost_history, W
alpha = 0.0001
iterations = 10000
cost_history, W = linearRegression(X, y, alpha, iterations)
X_test = X_data.values
y_test = y_data.values
test_cost = computeCost(X_test, y_test, W)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(len(cost_history)), cost_history, 'b', label='Training Loss')
ax.axhline(y=test_cost, color='r', linestyle='--', label='Testing Loss')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost', rotation=0)
ax.set_title('训练和损失曲线—alpha=0.0001')
plt.legend()
plt.show()
