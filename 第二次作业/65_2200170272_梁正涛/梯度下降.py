import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def configure_matplotlib():
    plt.rcParams.update({
        "mathtext.fontset": 'stix',
        "font.family": 'Arial Unicode MS',
        'axes.unicode_minus': False
    })

def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # 添加截距项
    X.insert(0, 'Intercept', 1)
    return X.values, y.values.reshape(-1, 1)

# 损失函数
def compute_cost(X, y, weights):
    predictions = np.dot(X, weights)
    errors = predictions - y
    return np.sum(errors ** 2) / (2 * len(X))

# 梯度下降
def gradient_descent(X, y, weights, alpha, iterations, X_test, y_test):
    train_cost_history = []
    test_cost_history = []  # 存储测试集损失历史
    for _ in range(iterations):
        predictions = np.dot(X, weights)
        errors = predictions - y
        updates = np.dot(X.T, errors) / len(X)
        weights -= alpha * updates
        train_cost = compute_cost(X, y, weights)
        test_cost = compute_cost(X_test, y_test, weights)  # 计算测试集损失
        train_cost_history.append(train_cost)
        test_cost_history.append(test_cost)
    return weights, train_cost_history, test_cost_history

# 主执行函数
def run_linear_regression(filepath, alpha=1e-8, iterations=10000):
    configure_matplotlib()
    X, y = load_and_prepare_data(filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    weights = np.zeros((X_train.shape[1], 1))
    weights, train_cost_history, test_cost_history = gradient_descent(X_train, y_train, weights, alpha, iterations, X_test, y_test)

    # 绘制训练集和测试集的损失曲线
    plt.plot(train_cost_history, label='训练集损失')
    plt.plot(test_cost_history, 'r--', label='测试集损失')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.title('训练集和测试集的损失曲线')
    plt.legend()
    plt.show()

    # 绘制回归线与实际值
    plt.scatter(X_train[:, 1], y_train, color='blue', label='训练集实际值')
    plt.scatter(X_test[:, 1], y_test, color='green', label='测试集实际值', alpha=0.5)
    plt.plot(X[:, 1], np.dot(X, weights), color='red', label='预测值')
    plt.xlabel('特征')
    plt.ylabel('目标值')
    plt.title('线性回归拟合')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    filepath = '/Users/liangzhengtao/Downloads/regress_data2.csv'
    run_linear_regression(filepath)