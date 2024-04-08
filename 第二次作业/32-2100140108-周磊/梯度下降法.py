import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
path = 'C:/Users/月初/Downloads/regress_data1.csv'
import pandas as pd

data = pd.read_csv(path)

# 分离特征和目标变量
cols = data.shape[1]
X_data = data.iloc[:, :cols - 1]
y_data = data.iloc[:, cols - 1:]

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=3)

# 在特征变量中添加偏置列
x_train = np.insert(x_train, 0, 1, axis=1)
x_test = np.insert(x_test, 0, 1, axis=1)

# 将特征变量和目标变量转换为numpy数组
X = x_train
Y = y_train.values

# 初始化参数W
W = np.array([[0.0], [0.0]])
print((X.shape, Y.shape, W.shape))


# 计算代价函数
def computeCost(X, Y, W, lambda_):
    m = X.shape[0]
    Y_hat = np.dot(X, W)
    loss = (1 / (2 * m)) * np.sum((Y_hat - Y) ** 2) + (lambda_ / (2 * m)) * np.sum(W ** 2)
    return loss


# 梯度下降法更新参数W
def gradientDescent(X, Y, W, alpha, lambda_):
    m = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = (1 / m) * np.dot(X.T, (Y_hat - Y)) + (lambda_ / m) * W
    W -= alpha * dW
    return W


# 线性回归训练函数
def linearRegression(X, Y, alpha, lambda_, iters):
    loss_history = []
    test_loss_history = []  # 存储测试损失历史记录
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss = computeCost(X, Y, W, lambda_)
        loss_history.append(loss)

        # 计算测试损失
        test_loss = computeCost(x_test, y_test.values, W, lambda_)
        test_loss_history.append(test_loss)

        W = gradientDescent(X, Y, W, alpha, lambda_)
    return loss_history, test_loss_history, W


# 预测函数
def predict(X, W):
    y_pre = np.dot(X, W)
    return y_pre


# 设置梯度下降的学习率和正则化参数
alpha = 0.0001
lambda_ = 0.1
iters = 10000

# 使用梯度下降法训练线性回归模型
loss_his, test_loss_his, W = linearRegression(X, Y, alpha, lambda_, iters)
print(W)

# 绘制预测结果
x = np.linspace(x_test[:, 1].min(), x_test[:, 1].max(), 100)
f = W[0, 0] + (W[1, 0] * x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(x_test[:, 1], y_test, label='测试数据')
ax.legend(loc=2)
ax.set_xlabel('人口')
ax.set_ylabel('收益', rotation=90)
ax.set_title('预测收益和人口规模')

# 绘制训练和测试损失曲线
plt.figure()
plt.plot(loss_his, label='训练损失')
plt.plot(test_loss_his, label='测试损失')  # 添加测试损失曲线
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('训练和测试损失曲线')
plt.legend()
plt.show()
