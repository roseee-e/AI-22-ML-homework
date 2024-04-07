import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

path = 'C:/Users/21155/Documents/Tencent Files/2115565902/FileRecv/regress_data1.csv'
data = pd.read_csv(path) ## data 是dataframe 的数据类型
data.head()

cols = data.shape[1]
X_data = data.iloc[:,:cols-1]#X是所有行，去掉最后一列， 未标准化
y_data = data.iloc[:,cols-1:]#X是所有行，最后一列
data.describe()

data.plot(kind='scatter', x='人口', y='收益', figsize=(4,3)) # 利用散点图可视化数据
plt.xlabel('人口')
plt.ylabel('收益', rotation=90)
plt.show()

X_data.insert(0, 'Ones', 1)
X_data.head()
y_data.head()

X=X_data.values
Y=y_data.values
W=np.array([[0.0],[0.0]]) ## 初始化W系数矩阵，w 是一个(2,1)矩阵
def computeCost(X, Y, W):
    Y_hat = np.dot(X,W)
    loss =np.sum((Y_hat - Y)** 2)/(2*X.shape[0])# (m,n) @ (n, 1) -> (n, 1)
    return loss
#%%
def computeCost1(X, Y, W):
    Y_hat = X@W
    loss =np.sum((Y_hat - Y)** 2)/(2*X.shape[0])# (m,n) @ (n, 1) -> (n, 1)
    return loss
(X.shape,Y.shape, W.shape)


def gradientDescent(X, Y, W, alpha):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = X.T @ (Y_hat - Y) / X.shape[0]
    #     dW = X.T@(Y_hat-Y)
    W += -alpha * dW
    return W


# %%
def linearRegression(X, Y, alpha, iters):
    loss_his = []
    # step1: initialize the model parameters
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))  ## 初始化W系数矩阵，w 是一个(feature_dim,1)矩阵
    ## repeat step 2 and step 3 untill to the convergence or the end of iterations
    for i in range(iters):
        # step2 : using the initilized parameters to predict the output and calculate the loss
        loss = computeCost(X, Y, W)
        loss_his.append(loss)
        # step3: using the gradient decent method to update the parameters
        W = gradientDescent(X, Y, W, alpha)
    return loss_his, W  ## 返回损失和模型参数。


# %%
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


# %%
alpha = 0.0001
iters = 10000
loss_his, W = linearRegression(X, Y, alpha, iters)

x = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(X_data['人口'], data['收益'], label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口' )
ax.set_ylabel('收益', rotation=90)
ax.set_title('预测收益和人口规模')
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('误差和训练Epoch数')
plt.show()

#2引入正则项后的模型
data = pd.read_csv(path)
X_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1]
# 插入一列全为1的向量
X_data.insert(0, 'Ones', 1)
# 转换为numpy数组
X = X_data.values
y = y_data.values.reshape(-1, 1)
# 计算代价函数，加入L2正则项
def computeCostRegularized(X, y, W, lamda):
    m = len(y)
    error = np.dot(X, W) - y
    cost = 1 / (2 * m) * np.sum(np.square(error)) + (lamda / (2 * m)) * np.sum(np.square(W[1:]))  # L2正则项
    return cost
# 梯度下降，加入L2正则项
def gradientDescentRegularized(X, y, W, alpha, lamda, iters):
    m = len(y)
    cost_history = np.zeros(iters)

    for i in range(iters):
        error = np.dot(X, W) - y
        W = W - (alpha / m) * (np.dot(X.T, error) + lamda * W)
        cost_history[i] = computeCostRegularized(X, y, W, lamda)

    return W, cost_history
alpha = 0.001
lamda = 0.1
iters = 10000
initial_W = np.zeros((X.shape[1], 1))
W, cost_history = gradientDescentRegularized(X, y, initial_W, alpha, lamda, iters)
x = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
f_regularized = W[0] + (W[1] * x)

plt.figure(figsize=(6, 4))
plt.scatter(X_data['人口'], y_data, label='训练数据')
plt.plot(x, f_regularized, 'r', label='引入正则项')
plt.xlabel('人口')
plt.ylabel('收益')
plt.legend()
plt.title('预测收益和人口规模-正则项系数为0.1]')

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), cost_history, 'r')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('误差和训练Epoch数')
plt.show()

#3最小二乘法求解线性回归模型
def least_squares(X, Y):
    W = np.linalg.inv(X.T @ X) @ X.T @ Y
    return W
# 计算最小二乘法的参数
W_least_squares = least_squares(X, Y)
print("最小二乘法求得的参数:\n", W_least_squares)
# 使用最小二乘法的参数进行预测
y_pred_train = predict(X, W_least_squares)
# 绘制训练数据的拟合曲线
plt.figure(figsize=(6, 4))
plt.plot(X[:,1], y_pred_train, 'r', label='拟合曲线')
plt.scatter(X[:,1], Y, label='训练数据')
plt.legend()
plt.xlabel('人口')
plt.ylabel('收益', rotation=90)
plt.title('最小二乘法拟合训练数据')
plt.show()


#4归一化的处理
data = pd.read_csv(path)

cols = data.shape[1]
X_data = data.iloc[:, :cols-1]
y_data = data.iloc[:, cols-1:]

X_data.insert(0, 'Ones', 1)

X = X_data.values
Y = y_data.values
W=np.array([[0.0],[0.0]])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)

def computeCost(X, Y, W):
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y)**2) / (2*X.shape[0])
    return loss

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


alpha = 0.003
iters = 10000
loss_his, W = linearRegression(X_scaled, Y_scaled, alpha, iters)

x_scaled = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
x_scaled = scaler.transform(x_scaled.reshape(-1, 1))
f_scaled = W[0, 0] + (W[1, 0] * x_scaled)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(scaler.inverse_transform(x_scaled), scaler.inverse_transform(f_scaled), 'r', label='预测值')
ax.scatter(X_data['人口'], data['收益'], label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口')
ax.set_ylabel('收益', rotation=90)
ax.set_title('归一化后的回归模型-alpha=0.003',color='red')

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('误差和训练Epoch数-alpha=0.003',color='red')
plt.show()

#训练与测试损失曲线
data = pd.read_csv(path)

cols = data.shape[1]
X_data = data.iloc[:, :cols-1]
y_data = data.iloc[:, cols-1:]

X_data.insert(0, 'Ones', 1)

X = X_data.values
Y = y_data.values

def computeCost(X, Y, W):
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y)**2) / (2*X.shape[0])
    return loss

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

alpha = 0.0001
iters = 10000
loss_train, W = linearRegression(X, Y, alpha, iters)

# Now let's calculate the test loss
X_test = X_data.values
Y_test = y_data.values
test_loss = computeCost(X_test, Y_test, W)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(len(loss_train)), loss_train, 'b', label='训练损失')
ax.axhline(y=test_loss, color='r', linestyle='--', label='测试损失')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('训练和测试损失曲线-alpha=0.0001',color='red')
plt.legend()
plt.show()