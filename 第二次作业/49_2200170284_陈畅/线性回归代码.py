import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams 
import pandas as pd
config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)  ## 设置画图的一些参数
## 读取数据
path = 'C:\Users\cycy20\Downloads\regress_data1.csv'
data = pd.read_csv(path) ## data 是dataframe 的数据类型
data.head() # 返回data中的前几行数据，默认是前5行。
cols = data.shape[1]
X_data = data.iloc[:,:cols-1]#X是所有行，去掉最后一列， 未标准化
y_data = data.iloc[:,cols-1:]#X是所有行，最后一列
X_data.insert(0, 'Ones', 1)
X_data.head()#head()是观察前5行
y_data.head()
X=X_data.values
Y=y_data.values
W=np.array([[0.0],[0.0]]) 
(X.shape,Y.shape, W.shape)
def computeCost(X, Y, W):
    Y_hat = np.dot(X,W)
    loss =np.sum((Y_hat - Y)** 2)/(2*X.shape[0])
    return loss
def computeCost1(X, Y, W):
    Y_hat = X@W
    loss =np.sum((Y_hat - Y)** 2)/(2*X.shape[0])
    return loss
def gradientDescent(X, Y, W, alpha):
    num_train = X.shape[0]
    Y_hat = np.dot(X,W)
    dW = X.T@(Y_hat-Y)/ X.shape[0]
    W += -alpha * dW
    return W
def linearRegression(X,Y, alpha, iters):
    loss_his = []
    feature_dim = X.shape[1]
    W=np.zeros((feature_dim,1)) ## 初始化W系数矩阵，w 是一个(feature_dim,1)矩阵
    for i in range (iters):
        loss = computeCost(X,Y,W)
        loss_his.append(loss)
        W=gradientDescent(X, Y, W, alpha)
    return loss_his, W ## 返回损失和模型参数。
def predict(X, W):
    y_pre = np.dot(X,W)
    return y_pre
alpha =0.0001
iters = 10000
loss_his, W = linearRegression(X,Y, alpha, iters)
x = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(X_data['人口'], data['收益'], label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口' )
ax.set_ylabel('收益', rotation=90)
ax.set_title('梯度下降')
# 计算代价函数，加入L2正则项
def computeCostRegularized(X, Y, W, lamda):
    m = len(Y)
    error = np.dot(X, W) - Y
    cost = 1 / (2 * m) * np.sum(np.square(error)) + (lamda / (2 * m)) * np.sum(np.square(W[1:]))  # L2正则项
    return cost
# 梯度下降，加入L2正则项
def gradientDescentRegularized(X, Y, W, alpha, lamda, iters):
    m = len(Y)
    cost_history = np.zeros(iters)
    for i in range(iters):
        error = np.dot(X, W) - Y
        W = W - (alpha / m) * (np.dot(X.T, error) + lamda * W)
        cost_history[i] = computeCostRegularized(X, Y, W, lamda)
    return W, cost_history
lamda = 0.1
initial_W = np.zeros((X.shape[1], 1))
W, cost_history = gradientDescentRegularized(X,Y, initial_W, alpha, lamda, iters)
# 引入正则项后的模型
x = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
f_regularized = W[0] + (W[1] * x)
plt.figure(figsize=(6, 4))
plt.scatter(X_data['人口'], y_data, label='训练数据')
plt.plot(x, f_regularized, 'r', label='引入正则项')
plt.xlabel('人口')
plt.ylabel('收益')
plt.legend()
plt.title('引入正则化')
X_data = (X_data - X_data.mean()) / X_data.std()
y_data = (y_data - y_data.mean()) / y_data.std()
# 提取特征和标签
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X = np.c_[np.ones(X.shape[0]), X]
W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
# 预测
y_pred = X.dot(W)
# 绘制结果
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 1], y, label='训练数据')
plt.plot(X[:, 1], y_pred, color='red', label='预测值')
plt.xlabel('人口')
plt.ylabel('收益')
plt.legend()
plt.title('最小二乘法求解线性回归模型')
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价')
ax.set_title('误差和训练Epoch数')
plt.show()


