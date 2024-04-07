import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import rcParams
  
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)  ## 设置画图的一些参数

# 读取数据
data = pd.read_csv('F:\\mycode\\python\\machine learning\\regress_data1.csv')

# 提取特征和目标变量
cols = data.shape[1]
X_data = data.iloc[:, :cols-1]  # X是所有行，去掉最后一列，未标准化
y_data = data.iloc[:, cols-1:]
X = data['人口'].values.reshape(-1, 1)
y = data['收益'].values.reshape(-1, 1)
W = np.array([[0.0], [0.0]])

# 在特征矩阵中添加一列全为1的偏置项
X = np.hstack((np.ones((X.shape[0], 1)), X))

# 定义计算损失的函数
def computeCost(X, Y, W):
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y)**2) / (2*X.shape[0])  # (m,n) @ (n, 1) -> (n, 1)
    return loss

# 定义梯度下降函数
def gradientDescent(X, Y, W, alpha):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = X.T @ (Y_hat - Y) / X.shape[0]
    W += -alpha * dW
    return W

def linearRegression(X, Y, alpha, iters):
    loss_his = []
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss = computeCost(X, Y, W)
        loss_his.append(loss)
        W = gradientDescent(X, Y, W, alpha)
    return loss_his, W

# 初始化模型参数和学习率
alpha = 0.0001
iters = 10000
loss_his, W = linearRegression(X, y, alpha, iters)

# 打印训练后的参数
print("训练后的参数：", W)

# 绘制训练过程中损失的变化曲线和预测值与数据点的散点图
plt.figure(figsize=(12, 4))

# 绘制训练过程中损失的变化曲线
plt.subplot(2, 2, 1)
plt.plot(range(1, iters + 1), loss_his, color='r')
plt.xlabel('迭代次数')
plt.ylabel('代价')
plt.title('误差和训练Epoch数')

# 绘制预测值和数据点的散点图
plt.subplot(2, 2, 2)
x = np.linspace(X.min(), X_data['人口'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)
plt.plot(x, f, 'r', label='预测值')
plt.scatter(X_data['人口'], data['收益'], label='训练数据')
plt.legend(loc=2)
plt.xlabel('人口')
plt.ylabel('收益')
plt.title('预测收益和人口规模')

# 训练和测试损失
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def linearRegression(X_train, y_train, X_test, y_test, alpha, iters):
    loss_train_his = []
    loss_test_his = []
    feature_dim = X_train.shape[1]
    W = np.zeros((feature_dim, 1)) 
    for i in range(iters):
        loss_train = computeCost(X_train, y_train, W)
        loss_train_his.append(loss_train)
        
        loss_test = computeCost(X_test, y_test, W)
        loss_test_his.append(loss_test)
        
        W = gradientDescent(X_train, y_train, W, alpha)
    return loss_train_his, loss_test_his, W

# 初始化模型参数和学习率
alpha = 0.0001
iters = 10000
loss_train_his, loss_test_his, W = linearRegression(X_train, y_train, X_test, y_test, alpha, iters)

# 打印训练后的参数
print("训练后的参数：", W)

# 绘制损失曲线
plt.subplot(2, 2, 3)
plt.plot(range(1, iters + 1), loss_train_his, label='训练损失', color='blue')
plt.plot(range(1, iters + 1), loss_test_his, label='测试损失', color='red')
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('训练和测试损失曲线')
plt.legend()


#l2
# 读取数据
data = pd.read_csv('F:\\mycode\\python\\machine learning\\regress_data1.csv')

# 提取特征和目标变量
X = data['人口'].values.reshape(-1, 1)
y = data['收益'].values.reshape(-1, 1)
W = np.array([[0.0], [0.0]])

# 在特征矩阵中添加一列全为1的偏置项
X = np.hstack((np.ones((X.shape[0], 1)), X))

# 定义计算损失的函数
def computeCost(X, Y, W, lambda_reg):
    m = X.shape[0]
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y)**2) / (2 * m)
    regularization_term = (lambda_reg / (2 * m)) * np.sum(W[1:]**2)  # 排除偏置项进行正则化
    return loss + regularization_term

# 定义梯度下降函数
def gradientDescent(X, Y, W, alpha, lambda_reg):
    m = X.shape[0]
    Y_hat = np.dot(X, W)
    regularized_term = (lambda_reg / m) * W[1:]  # 排除偏置项进行正则化
    dW = (1 / m) * np.dot(X.T, (Y_hat - Y)) + np.vstack((0, regularized_term))  # 在梯度中包含正则化项
    W -= alpha * dW
    return W

def linearRegressionWithRegularization(X, Y, alpha, lambda_reg, iters):
    loss_his = []
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss = computeCost(X, Y, W, lambda_reg)
        loss_his.append(loss)
        W = gradientDescent(X, Y, W, alpha, lambda_reg)
    return loss_his, W

# 初始化模型参数和学习率
alpha = 0.0001
lambda_reg = 0.1  # 正则化参数
iters = 10000

# 使用L2范数正则项的梯度下降法进行线性回归
loss_his, W = linearRegressionWithRegularization(X, y, alpha, lambda_reg, iters)

# 打印训练后的参数
print("训练后的参数：", W)

# 绘制预测值和数据点的散点图
plt.subplot(2, 2, 4)
plt.scatter(X[:,1], y, label='训练数据')
x = np.linspace(X.min(), X.max(), 100)
f = W[0, 0] + (W[1, 0] * x)
plt.plot(x, f, 'r', label='预测值')
plt.xlabel('人口')
plt.ylabel('收益')
plt.title('带L2范数正则项的线性回归')
plt.legend()
plt.show()
