import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 设置Matplotlib绘图配置
def set_matplotlib_config():
    config = {
        "mathtext.fontset": 'stix',
        "font.family": 'serif',
        "font.serif": ['SimHei'],
        "font.size": 10,
        'axes.unicode_minus': False
    }
    plt.rcParams.update(config)

# 数据归一化处理
def normalize_data(X):
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, scaler

# 计算损失函数
def compute_cost(X, Y, W):
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y) ** 2) / (2 * len(X))
    return loss

# 梯度下降算法
def gradient_descent(X, Y, W, alpha, lambda_):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = X.T @ (Y_hat - Y) / num_train
    W = (1 - alpha * lambda_ / num_train) * W - alpha * dW
    return W

# 线性回归模型
def linear_regression(X, Y, alpha, iters, lambda_):
    loss_history = []
    W = np.zeros((X.shape[1], 1))
    for i in range(iters):
        W = gradient_descent(X, Y, W, alpha, lambda_)
        loss = compute_cost(X, Y, W)
        loss_history.append(loss)
    return loss_history, W

# 预测函数
def predict(X, W):
    return np.dot(X, W)

# 最小二乘法求解线性回归模型
def least_squares(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y

# 主程序
def main():
    set_matplotlib_config()
    data = pd.read_csv(r'D:\python\regress_data1.csv')
    X_data = data.iloc[:, :-1].values
    y_data = data.iloc[:, -1].values.reshape(-1, 1)

    # 数据归一化
    X_data_norm, scaler = normalize_data(X_data)
    X_data_norm = np.insert(X_data_norm, 0, 1, axis=1)

    # 线性回归
    alpha = 0.001
    lambda_ = 0.001
    iters = 20000
    loss_history, W = linear_regression(X_data_norm, y_data, alpha, iters, lambda_)
    print("训练后的参数:\n", W)

    # 可视化训练结果
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X_data_norm[:, 1], y_data, color='green', label='训练数据')
    plt.plot(X_data_norm[:, 1], predict(X_data_norm, W), color='blue', label='预测值')
    plt.xlabel('人口')
    plt.ylabel('收益', rotation=90)
    plt.title('预测收益和人口规模')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(iters), loss_history, color='blue')
    plt.xlabel('迭代次数')
    plt.ylabel('损失', rotation=0)
    plt.title('误差和训练Epoch数')
    plt.tight_layout()
    plt.show()

    # 使用最小二乘法求解线性回归模型
    W_least_squares = least_squares(X_data_norm, y_data)
    print("最小二乘法求得的参数:\n", W_least_squares)

    # 可视化最小二乘法结果
    plt.figure(figsize=(6, 4))
    plt.scatter(X_data_norm[:, 1], y_data, color='green', label='训练数据')
    plt.plot(X_data_norm[:, 1], predict(X_data_norm, W_least_squares), color='blue', label='拟合曲线')
    plt.xlabel('人口')
    plt.ylabel('收益', rotation=90)
    plt.title('最小二乘法拟合训练数据')
    plt.legend()
    plt.show()

    # 测试数据预处理和预测
    test_data = np.array([
         [5.6063, 3.3928],
    [12.836, 10.117],
    [6.3534, 5.4974],
    [5.4069, 0.55657],
    [6.8825, 3.9115],
    [11.708, 5.3854],
    [5.7737, 2.4406],
    [7.8247, 6.7318],
    [7.0931, 1.0463],
    [5.0702, 5.1337],
    [5.8014, 1.844],
    [11.7, 8.0043],
    [5.5416, 1.0179],
    [7.5402, 6.7504],
    [5.3077, 1.8396],
    [7.4239, 4.2885],
    [7.6031, 4.9981],
    [6.3328, 1.4233],
    [6.3589, -1.4211],
    [6.2742, 2.4756],
    [5.6397, 4.6042],
    [9.3102, 3.9624],
    [9.4536, 5.4141],
    [8.8254, 5.1694],
    [5.1793, -0.74279],
    [21.279, 17.929],
    [14.908, 12.054],
    [18.959, 17.054],
    [7.2182, 4.8852],
    [8.2951, 5.7442],
    [10.236, 7.7754]
    ])
    X_test = test_data[:, 0].reshape(-1, 1)
    Y_test = test_data[:, 1].reshape(-1, 1)
    X_test_norm = scaler.transform(X_test)
    X_test_norm = np.insert(X_test_norm, 0, 1, axis=1)
    test_loss_history = [compute_cost(X_test_norm, Y_test, W) for _ in range(iters)]

    # 可视化训练和测试损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color='blue', label='Training Loss')
    plt.plot(test_loss_history, color='green', label='Test Loss')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.title('训练和测试损失曲线')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()