import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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
    m = len(Y)
    Y_hat = np.dot(X, W)
    cost = (1 / (2 * m)) * np.sum((Y_hat - Y) ** 2)
    return cost


# 梯度下降算法
def gradient_descent(X, Y, W, alpha, iters, lambda_=0):
    m = len(Y)
    cost_history = []

    for i in range(iters):
        Y_hat = np.dot(X, W)
        loss = Y_hat - Y
        gradient = np.dot(X.T, loss) / m + (lambda_ / m) * W
        W = W - alpha * gradient
        cost = compute_cost(X, Y, W)
        cost_history.append(cost)

    return W, cost_history


# 线性回归模型
def linear_regression(X, Y, alpha=0.01, iters=1000, lambda_=0):
    X_b = np.c_[np.ones((len(X), 1)), X]  # 添加x0 = 1
    W = np.random.randn(X_b.shape[1], 1)
    W, cost_history = gradient_descent(X_b, Y, W, alpha, iters, lambda_)

    return W, cost_history


# 最小二乘法
def least_squares(X, Y):
    X_b = np.c_[np.ones((len(X), 1)), X]  # 添加x0 = 1
    W = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
    return W


# 预测函数
def predict(X, W):
    X_b = np.c_[np.ones((len(X), 1)), X]  # 添加x0 = 1
    return np.dot(X_b, W)


# 绘制并保存训练与测试损失曲线
def plot_and_save_loss_curve(train_loss, test_loss, filename="training_and_test_loss_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='训练损失')
    plt.plot(test_loss, label='测试损失', linestyle='--')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.title('训练与测试损失曲线')
    plt.legend()
    plt.savefig(filename)
    plt.close()  # 关闭图形，避免重复显示


# 绘制并保存L2正则化回归效果
def plot_and_save_regression_results(X, Y, Y_pred, filename="L2_regularization_regression_effect.png"):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, color='green', label='实际数据')
    plt.plot(X, Y_pred, color='blue', label='预测值')
    plt.xlabel('人口')
    plt.ylabel('收益')
    plt.title('L2正则化回归效果')
    plt.legend()
    plt.savefig(filename)
    plt.close()


# 主程序
def main():
    set_matplotlib_config()
    # 请将路径替换为您自己的数据文件路径
    data_path = r"C:/Users/小黑/Desktop/机器学习/第二次作业/26_2200170246_莫永清/regress_data1.csv"
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1, 1)

    # 数据归一化
    X_norm, scaler = normalize_data(X)

    # 分割数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=0.2, random_state=42)

    # 训练模型
    W, cost_history = linear_regression(X_train, Y_train, alpha=0.01, iters=2000, lambda_=0)
    W_least_squares = least_squares(X_train, Y_train)

    # 预测
    Y_pred = predict(X_test, W)
    Y_pred_least_squares = predict(X_test, W_least_squares)

    # 计算MSE
    mse = mean_squared_error(Y_test, Y_pred)
    mse_least_squares = mean_squared_error(Y_test, Y_pred_least_squares)
    print("线性回归MSE:", mse)
    print("最小二乘法MSE:", mse_least_squares)

    # 绘制并保存L2正则化回归效果
    plot_and_save_regression_results(X_test, Y_test, Y_pred, "L2_regularization_regression_effect.png")

    # 绘制训练和测试损失曲线
    plot_and_save_loss_curve(cost_history, cost_history, "training_and_test_loss_curve.png")

    # 绘制并保存最小二乘法图像
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, Y_test, color='green', label='实际数据')
    plt.plot(X_test, Y_pred_least_squares, color='blue', label='预测值')
    plt.xlabel('人口')
    plt.ylabel('收益')
    plt.title('最小二乘法')
    plt.legend()
    plt.savefig("least_squares_regression_effect.png")
    plt.close()


if __name__ == "__main__":
    main()，