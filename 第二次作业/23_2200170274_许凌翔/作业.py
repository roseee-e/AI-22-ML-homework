import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 设置Matplotlib绘图配置
def set_matplotlib_config():
    config = {
        "mathtext.fontset": 'stix',  # 使用STIX字体
        "font.family": 'serif',  # 使用衬线字体
        "font.serif": ['SimHei'],  # 使用黑体作为serif字体
        "font.size": 10,  # 字体大小设置为10
        'axes.unicode_minus': False  # 正确显示负号
    }
    plt.rcParams.update(config)

# 数据归一化处理
def normalize_data(X):
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)  # 归一化X数据
    return X_norm, scaler

# 计算损失函数
def compute_cost(X, Y, W):
    m = len(Y)
    Y_hat = np.dot(X, W)  # 预测值
    cost = (1 / (2 * m)) * np.sum(np.square(Y_hat - Y))  # 计算均方误差损失
    return cost

# 梯度下降算法
def gradient_descent(X, Y, W, alpha, iters, lambda_=0):
    m = len(Y)
    cost_history = []
    
    for i in range(iters):
        Y_hat = np.dot(X, W)  # 计算预测值
        loss = Y_hat - Y  # 计算误差
        # 计算梯度，注意不对W[0]即偏置项进行正则化
        gradient = (np.dot(X.T, loss) + lambda_ * np.r_[np.zeros([1, W.shape[1]]), W[1:]]) / m
        W = W - alpha * gradient  # 更新权重
        cost = compute_cost(X, Y, W)  # 计算当前损失
        cost_history.append(cost)  # 保存损失历史
        
    return W, cost_history

# 线性回归模型
def linear_regression(X, Y, alpha=0.01, iters=50000, lambda_=0):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 在X数据前加上一列1作为偏置项
    W = np.zeros((X_b.shape[1], 1))  # 权重初始化为0
    W, cost_history = gradient_descent(X_b, Y, W, alpha, iters, lambda_)  # 应用梯度下降
    
    return W, cost_history
# 最小二乘法拟合
def least_squares(X, Y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 在X数据前加上一列1作为偏置项
    W_ls = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)  # 最小二乘法公式
    return W_ls

# 预测函数
def predict(X, W):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 在X数据前加上一列1作为偏置项
    Y_pred = X_b.dot(W)  # 使用权重进行预测
    return Y_pred
# 线性回归模型
def linear_regression(X, Y, alpha=0.01, iters=50000, lambda_=0):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 在X数据前加上一列1作为偏置项
    W = np.zeros((X_b.shape[1], 1))  # 权重初始化为0
    cost_history = []  # 保存损失历史
    
    for i in range(iters):
        Y_hat = np.dot(X_b, W)  # 计算预测值
        loss = Y_hat - Y  # 计算误差
        gradient = (np.dot(X_b.T, loss) + lambda_ * np.r_[np.zeros([1, W.shape[1]]), W[1:]]) / len(Y)  # 计算梯度
        W = W - alpha * gradient  # 更新权重
        cost = compute_cost(X_b, Y, W)  # 计算当前损失
        cost_history.append(cost)  # 保存损失历史
        
    return W, cost_history
# 绘制并保存所有图像
def plot_all_graphs(train_loss, test_loss, X_train, Y_train, Y_train_pred, X_test, Y_test, Y_test_pred):
    plt.figure(figsize=(16, 12))

    # 训练与测试损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_loss, label='训练损失', color='blue')
    plt.plot(test_loss, label='测试损失', linestyle='--', color='orange')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.title('训练与测试损失曲线')
    plt.legend()

    # 训练数据散点图和拟合线
    plt.subplot(2, 2, 2)
    plt.scatter(X_train, Y_train, color='blue', label='训练数据')
    plt.plot(X_train, Y_train_pred, color='red', label='拟合线')
    plt.xlabel('人口')
    plt.ylabel('收益')
    plt.title('训练数据拟合效果')
    plt.legend()

    # 测试数据散点图和拟合线
    plt.subplot(2, 2, 3)
    plt.scatter(X_test, Y_test, color='blue', label='测试数据')
    plt.plot(X_test, Y_test_pred, color='red', label='拟合线')
    plt.xlabel('人口')
    plt.ylabel('收益')
    plt.title('测试数据拟合效果')
    plt.legend()

    # 最小二乘法拟合数据
    plt.subplot(2, 2, 4)
    W_ls = least_squares(X_train, Y_train)
    Y_pred_ls = predict(X_train, W_ls)
    plt.scatter(X_train, Y_train, color='blue', label='数据散点')
    plt.plot(X_train, Y_pred_ls, color='red', label='最小二乘法拟合线')
    plt.xlabel('人口')
    plt.ylabel('收益')
    plt.title('最小二乘法拟合数据')
    plt.legend()

    # 调整布局
    plt.tight_layout()

    # 显示图像
    plt.show()

# 主程序
def main():
    set_matplotlib_config()
    
    # 读取数据，确保已经将文件路径改为正确的路径
    data_path = r"C:\Users\小黑\Desktop\机器学习\第二次作业\26_2200170246_莫永清\regress_data1.csv"
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1, 1)

    # 数据归一化处理
    X_norm, scaler = normalize_data(X)
    
    # 分割数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=0.2, random_state=42)
    
    # 训练模型并计算训练和测试损失
    W, train_cost_history = linear_regression(X_train, Y_train, alpha=0.01, iters=50000, lambda_=1)
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]  # 测试数据加上偏置项
    test_cost_history = [compute_cost(X_test_b, Y_test, W) for _ in range(50000)]  # 计算测试损失

    # 计算训练数据的拟合值
    Y_train_pred = np.dot(np.c_[np.ones((X_train.shape[0], 1)), X_train], W)
    # 计算测试数据的拟合值
    Y_test_pred = np.dot(X_test_b, W)

    # 绘制所有图像
    plot_all_graphs(train_cost_history, test_cost_history, X_train, Y_train, Y_train_pred, X_test, Y_test, Y_test_pred)

if __name__ == "__main__":
    main()

# 主程序
def main():
    set_matplotlib_config()
    
    # 读取数据，确保已经将文件路径改为正确的路径
    data_path = r"C:\Users\小黑\Desktop\机器学习\第二次作业\26_2200170246_莫永清\regress_data1.csv"
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1, 1)

    # 数据归一化处理
    X_norm, scaler = normalize_data(X)
    
    # 分割数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=0.2, random_state=42)
    
    # 训练模型并计算训练和测试损失
    W, train_cost_history = linear_regression(X_train, Y_train, alpha=0.01, iters=50000, lambda_=1)
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]  # 测试数据加上偏置项
    test_cost_history = [compute_cost(X_test_b, Y_test, W) for _ in range(50000)]  # 计算测试损失

    # 计算训练数据的拟合值
    Y_train_pred = np.dot(np.c_[np.ones((X_train.shape[0], 1)), X_train], W)
    # 计算测试数据的拟合值
    Y_test_pred = np.dot(X_test_b, W)

    # 绘制所有图像
    plot_all_graphs(train_cost_history, test_cost_history, X_train, Y_train, Y_train_pred, X_test, Y_test, Y_test_pred)

    # 绘制训练误差随Epoch数的变化
    plt.figure()
    plt.plot(range(len(train_cost_history)), train_cost_history, color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.show()

if __name__ == "__main__":
    main()