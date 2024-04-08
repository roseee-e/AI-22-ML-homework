import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
from matplotlib import rcParams  # run command settings for plotting

config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)  # 设置画图的一些参数

#  读取数据
path = 'D:\\QQ缓存文件\\regress_data1.csv'
data = pd.read_csv(path)  # data 是dataframe 的数据类型
data.head()  # 返回data中的前几行数据，默认是前5行。
# print(data.head())

cols = data.shape[1]
X_data = data.iloc[:, :cols-1]  # X是所有行，去掉最后一列， 未标准化
y_data = data.iloc[:, cols-1:]  # X是所有行，最后一列

data.describe()  # 查看数据的统计信息
# print(data.describe())

data.plot(kind='scatter', x='人口', y='收益', figsize=(4, 3))  # 利用散点图可视化数据
plt.xlabel('人口')
plt.ylabel('收益', rotation=90)

X_data.insert(0, 'Ones', 1)

X_data.head()

y_data.head()

X = X_data.values
Y = y_data.values
W = np.array([[0.0], [0.0]])  # 初始化W系数矩阵，w 是一个(2,1)矩阵

# print((X.shape, Y.shape, W.shape))  # 看下维度

def computeCost(X, Y, W):
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y) ** 2)/(2*X.shape[0])  # (m,n) @ (n, 1) -> (n, 1)
    return loss

def computeCost1(X, Y, W):
    Y_hat = X@W
    loss = np.sum((Y_hat - Y) ** 2)/(2*X.shape[0])  # (m,n) @ (n, 1) -> (n, 1)
    return loss

def gradientDescent(X, Y, W, alpha):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = X.T@(Y_hat-Y) / X.shape[0]
#     dW = X.T@(Y_hat-Y)
    W += -alpha * dW
    return W


def linearRegression(X, Y, alpha, iters):
    loss_his = []
    # step1: initialize the model parameters
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))  # 初始化W系数矩阵，w 是一个(feature_dim,1)矩阵
    # repeat step 2 and step 3 untill to the convergence or the end of iterations
    for i in range(iters):
        # step2 : using the initilized parameters to predict the output and calculate the loss
        loss = computeCost(X, Y, W)
        loss_his.append(loss)
        # step3: using the gradient decent method to update the parameters
        W = gradientDescent(X, Y, W, alpha)
    return loss_his, W  # 返回损失和模型参数。

def predict(X, W):
    '''
    输入：
        X：测试数据集
        W：模型训练好的参数
    输出：
        y_pre：预测值
    '''
    y_pre = np.dot(X, W)
    return y_pre

alpha = 0.0001
iters = 10000
lamda = 0.01
loss_his, W = linearRegression(X, Y, alpha, iters)

# L2z

x = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(X_data['人口'], data['收益'], label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口')
ax.set_ylabel('收益', rotation=90)
ax.set_title('预测收益和人口规模')
# plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('误差和训练Epoch数')
# 设置Matplotlib绘图配置
def set_matplotlib_config():
    config = {
        "mathtext.fontset": 'stix',
        "font.family": 'serif',
        "font.serif": ['SimHei'],
        "font.size": 10,  # 字号，大家自行调节
        'axes.unicode_minus': False  # 处理负号，即-号
    }
    plt.rcParams.update(config)  # 设置画图的一些参数

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
def gradient_descent(X, Y, W, alpha, iters, lambda_):
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

alpha = 0.01
iters = 1000
lambda_ = 0
# 线性回归模型
def linear_regression(X, Y, alpha, iters, lambda_):
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
def plot_and_save_loss_curve(train_loss, test_loss, filename="训练与测试损失曲线.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='训练损失')
    plt.plot(test_loss, label='测试损失', linestyle='-.')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.title('训练与测试损失曲线')
    plt.legend()
    plt.savefig(filename)
    plt.close()  # 关闭图形，避免重复显示


# 绘制并保存L2正则化回归效果
def plot_and_save_regression_results(X, Y, Y_pred, filename="L2正则化回归效果.png"):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, color='red', label='实际数据')
    plt.plot(X, Y_pred, color='black', label='预测值')
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
    data_path = r"D:\\QQ缓存文件\\regress_data1.csv"
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
    plot_and_save_regression_results(X_test, Y_test, Y_pred, "L2正则化回归效果.png")

    # 绘制训练和测试损失曲线
    plot_and_save_loss_curve(cost_history, cost_history, "训练与测试损失曲线.png")

    # 绘制并保存最小二乘法图像
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, Y_test, color='red', label='实际数据')
    plt.plot(X_test, Y_pred_least_squares, color='black', label='预测值')
    plt.xlabel('人口')
    plt.ylabel('收益')
    plt.title('最小二乘法')
    plt.legend()
    plt.savefig("最小二乘法.png")
    plt.close()

if __name__ == "__main__":
    main()

