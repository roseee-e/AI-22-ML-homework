import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
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
X_data = data.iloc[:, :cols - 1]  # 特征变量
y_data = data.iloc[:, cols - 1:]  # 目标变量

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=3)
# 在特征变量中添加偏置列
x_train = np.insert(x_train, 0, 1, axis=1)
x_test = np.insert(x_test, 0, 1, axis=1)

# 将特征变量和目标变量转换为numpy数组
X = x_train
Y = y_train.values

# 初始化参数W
W = np.array([[0.0], [0.0]])  ## 初始化W系数矩阵，w 是一个(2,1)矩阵
print((X.shape, Y.shape, W.shape))


# 计算代价函数
def computeCost(X, Y, W, lambda_):
    m = X.shape[0]
    Y_hat = np.dot(X, W)
    loss = (1 / (2 * m)) * np.sum((Y_hat - Y) ** 2) + (lambda_ / (2 * m)) * np.sum(W ** 2)
    return loss


# 使用最小二乘法求解线性回归参数
def linearRegression_least_squares(X, Y):
    W = np.linalg.lstsq(X, Y, rcond=None)[0]
    return W


# 预测函数
def predict(X, W):
    y_pre = np.dot(X, W)
    return y_pre


# 使用最小二乘法计算参数并绘制预测结果
W_least_squares = linearRegression_least_squares(X, Y)

# 计算训练和测试损失
train_loss = computeCost(X, Y, W_least_squares, lambda_=0)
test_loss = computeCost(x_test, y_test.values, W_least_squares, lambda_=0)

print("训练损失:", train_loss)
print("测试损失:", test_loss)

# 画出训练和测试损失曲线
num_iterations = 1000
learning_rate = 0.01
lambda_ = 0
train_losses = []
test_losses = []
for i in range(num_iterations):
    # 更新参数W
    Y_hat = np.dot(X, W)
    W -= learning_rate * (1 / X.shape[0]) * (np.dot(X.T, (Y_hat - Y)) + lambda_ * W)

    # 计算训练和测试损失并记录
    train_loss = computeCost(X, Y, W, lambda_)
    test_loss = computeCost(x_test, y_test.values, W, lambda_)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

# 绘制训练和测试损失曲线
plt.plot(range(num_iterations), train_losses, label='训练损失')
plt.plot(range(num_iterations), test_losses, label='测试损失')
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.legend()
plt.title('训练和测试损失曲线')
plt.show()

# 绘制预测结果
x = np.linspace(x_test[:, 1].min(), x_test[:, 1].max(), 100)
f = W_least_squares[0, 0] + (W_least_squares[1, 0] * x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(x_test[:, 1], y_test, label='测试数据')
ax.legend(loc=2)
ax.set_xlabel('人口')
ax.set_ylabel('收益', rotation=90)
ax.set_title('预测收益和人口规模（最小二乘法）')
plt.show()
