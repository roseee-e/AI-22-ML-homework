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
    X.insert(0, 'Intercept', 1)
    return X.values, y.values.reshape(-1, 1)


def solve_via_least_squares(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def run_linear_regression(filepath):
    configure_matplotlib()
    X, y = load_and_prepare_data(filepath)
    weights = solve_via_least_squares(X, y)

    # 分割数据集为了绘制和检验
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 绘制训练集和测试集的实际值
    plt.scatter(X_train[:, 1], y_train, color='blue', label='训练集实际值')
    plt.scatter(X_test[:, 1], y_test, color='green', label='测试集实际值', alpha=0.5)

    # 绘制回归线
    x_values = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
    y_values = weights[0] + weights[1] * x_values
    plt.plot(x_values, y_values, color='red', label='预测值')

    plt.xlabel('特征')
    plt.ylabel('目标值')
    plt.title('最小二乘法线性回归拟合')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    filepath = '/Users/liangzhengtao/Downloads/regress_data2.csv'
    run_linear_regression(filepath)