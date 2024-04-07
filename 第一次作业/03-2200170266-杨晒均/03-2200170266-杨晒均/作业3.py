
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
def load_data(path):
    data = pd.read_csv(path)
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    features.insert(0, 'Ones', 1)
    return features, labels
def compute_cost_regularized(features, labels, weights, lambda_):
    m = len(labels)
    error = np.dot(features, weights) - labels
    cost = 1 / (2 * m) * np.sum(np.square(error)) + (lambda_ / (2 * m)) * np.sum(np.square(weights[1:]))
    return cost
def gradient_descent_regularized(features, labels, weights, alpha, lambda_, iterations):
    m = len(labels)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):        
        error = np.dot(features, weights) - labels
        weights = weights - (alpha / m) * (np.dot(features.T, error) + lambda_ * weights)
        cost_history[i] = compute_cost_regularized(features, labels, weights, lambda_)
    
    return weights, cost_history
def plot_data_regression(features, labels, weights):
    x = np.linspace(features['人口'].min(), features['人口'].max(), 100)
    f_regularized = weights[0] + (weights[1] * x)
    plt.figure(figsize=(6, 4))
    plt.scatter(features['人口'], labels, label='训练数据')
    plt.plot(x, f_regularized, 'r', label='引入正则项')
    plt.xlabel('人口')
    plt.ylabel('收益')
    plt.legend()
    plt.title('预测收益以及人口规模')
    plt.show()
def plot_cost_history(cost_history, iterations):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(iterations), cost_history, 'r')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('代价', rotation=0)
    ax.set_title('误差以及训练Epoch数')
    plt.show()
# 设置绘图参数
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)
# 读取数据
data_path = 'C:/Users/杨晒均/Documents/Tencent Files/2164528672/FileRecv/regress_data1.csv'
features, labels = load_data(data_path)

alpha = 0.001
lambda_ = 0.1
iterations = 10000
initial_weights = np.zeros((features.shape[1], 1))
# 梯度下降
weights, cost_history = gradient_descent_regularized(features.values, labels.values.reshape(-1, 1),
                                                     initial_weights, alpha, lambda_, iterations)
# 绘制数据和回归线
plot_data_regression(features, labels, weights)
# 绘制代价
plot_cost_history(cost_history, iterations)

