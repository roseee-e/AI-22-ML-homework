
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams  ## run command settings for plotting
config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)  ## 设置画图的一些参数
## 读取数据
path = r'E:\机器学习\第二次实验\59-2200170292-黄家福\regress_data1.csv'

import pandas as pd
data = pd.read_csv(path) ## data 是dataframe 的数据类型
data.head() # 返回data中的前几行数据，默认是前5行。

cols = data.shape[1]
X_data = data.iloc[:,:cols-1]#X是所有行，去掉最后一列， 未标准化
y_data = data.iloc[:,cols-1:]#X是所有行，最后一列

data.describe() ## 查看数据的统计信息
data.plot(kind='scatter', x='人口', y='收益', figsize=(4,3)) # 利用散点图可视化数据
import matplotlib
plt.xlabel('人口')
plt.ylabel('收益', rotation=90)
plt.show()
X_data.insert(0, 'Ones', 1)
X_data.head()#head()是观察前5行
y_data.head()
X=X_data.values
Y=y_data.values
W=np.array([[0.0],[0.0]]) ## 初始化W系数矩阵，w 是一个(2,1)矩阵
(X.shape,Y.shape, W.shape)
def computeCost(X, Y, W):
    Y_hat = np.dot(X,W)
    loss =np.sum((Y_hat - Y)** 2)/(2*X.shape[0])# (m,n) @ (n, 1) -> (n, 1)
    return loss
def computeCost1(X, Y, W):
    Y_hat = X@W
    loss =np.sum((Y_hat - Y)** 2)/(2*X.shape[0])# (m,n) @ (n, 1) -> (n, 1)
    return loss
def gradientDescent(X, Y, W, alpha):
    num_train = X.shape[0]
    Y_hat = np.dot(X,W)
    dW = X.T@(Y_hat-Y)/ X.shape[0]
#     dW = X.T@(Y_hat-Y)
    W += -alpha * dW
    return W
def linearRegression(X,Y, alpha, iters):
    loss_his = []
    # step1: initialize the model parameters
    feature_dim = X.shape[1]
    W=np.zeros((feature_dim,1)) ## 初始化W系数矩阵，w 是一个(feature_dim,1)矩阵
    ## repeat step 2 and step 3 untill to the convergence or the end of iterations
    for i in range (iters):
        # step2 : using the initilized parameters to predict the output and calculate the loss
        loss = computeCost(X,Y,W)
        loss_his.append(loss)
        # step3: using the gradient decent method to update the parameters
        W=gradientDescent(X, Y, W, alpha)
    return loss_his, W ## 返回损失和模型参数。
def predict(X, W):
    '''
    输入：
        X：测试数据集
        W：模型训练好的参数
    输出：
        y_pre：预测值
    '''
    y_pre = np.dot(X,W)
    return y_pre
alpha =0.0001
iters = 10000
loss_his, W = linearRegression(X,Y, alpha, iters)
np.array([[-0.57602166],
       [ 0.85952782]])
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



# 2
# 引入L2范数正则项，观察回归效果的变化

def computeCost_L2(X, Y, W, lambda_reg):
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y) ** 2) / (2 * X.shape[0]) + (lambda_reg / (2 * X.shape[0])) * np.sum(W**2)
    return loss

def gradientDescent_L2(X, Y, W, alpha, lambda_reg):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = (X.T @ (Y_hat - Y) + lambda_reg * W) / X.shape[0]
    W += -alpha * dW
    return W


def computeCost_L2(X, Y, W, lambda_reg):
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y) ** 2) / (2 * X.shape[0]) + (lambda_reg / (2 * X.shape[0])) * np.sum(W**2)
    return loss

def gradientDescent_L2(X, Y, W, alpha, lambda_reg):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = (X.T @ (Y_hat - Y) + lambda_reg * W) / X.shape[0]
    W += -alpha * dW
    return W

alpha = 0.0001
iters = 10000
lambda_reg = 0.1  # 设置L2正则化参数

def linearRegression_L2(X, Y, alpha, iters, lambda_reg):
    loss_his = []
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))

    for i in range(iters):
        loss = computeCost_L2(X, Y, W, lambda_reg)
        loss_his.append(loss)
        W = gradientDescent_L2(X, Y, W, alpha, lambda_reg)

    return loss_his, W

loss_his_L2, W_L2 = linearRegression_L2(X, Y, alpha, iters, lambda_reg)

# 绘制损失曲线
plt.figure(figsize=(6, 4))
plt.plot(np.arange(iters), loss_his_L2, 'r')
plt.xlabel('迭代次数')
plt.ylabel('代价', rotation=0)
plt.title('带L2正则项的误差和训练Epoch数')
plt.show()




# 3
# 利用最小二乘法求解线性回归模型

def linearRegression_OLS(X, Y):
    W = np.linalg.inv(X.T @ X) @ X.T @ Y
    return W
import numpy as np
import matplotlib.pyplot as plt

# 生成一些随机数据作为示例
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)

# 使用最小二乘法计算线性回归模型的参数
X_b = np.c_[np.ones((100, 1)), X]  # 在X矩阵中添加一列1，用于计算截距
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)

# 绘制数据点
plt.scatter(X, Y)

# 绘制拟合直线
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
Y_predict = X_new_b.dot(theta_best)
plt.plot(X_new, Y_predict, 'r-')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('使用最小二乘法拟合的线性回归模型')
plt.show()


# 4
# 引入数据归一化，观察回归结果的变化

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
import numpy as np
import matplotlib.pyplot as plt

# 生成一些随机数据作为示例
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)

# 数据归一化
X_normalized = (X - np.mean(X)) / np.std(X)

# 使用最小二乘法计算归一化后的线性回归模型的参数
X_b = np.c_[np.ones((100, 1)), X_normalized]  # 在归一化后的X矩阵中添加一列1，用于计算截距
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)

# 绘制数据点
plt.scatter(X_normalized, Y)

# 绘制归一化后的拟合直线
X_new = np.array([[-2], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
X_new_normalized = (X_new - np.mean(X)) / np.std(X)
X_new_b_normalized = np.c_[np.ones((2, 1)), X_new_normalized]
Y_predict = X_new_b_normalized.dot(theta_best)
plt.plot(X_new_normalized, Y_predict, 'r-')

plt.xlabel('Normalized X')
plt.ylabel('Y')
plt.title('使用归一化后的数据拟合的线性回归模型')
plt.show()



# 5
#画出训练和测试损失曲线

def linearRegression(X, Y, alpha, iters):
    loss_his = []
    for i in range(iters):
        loss = computeCost(X, Y, W)
        loss_his.append(loss)
        W = gradientDescent(X, Y, W, alpha)
    return loss_his, W
from sklearn.model_selection import train_test_split

# 划分数据集为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 对训练集和测试集进行预处理
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 插入偏置列
X_train_b = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_b = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

def computeTestLoss(X_test, Y_test, W):
    Y_hat_test = np.dot(X_test, W)
    test_loss = np.sum((Y_hat_test - Y_test) ** 2) / (2 * X_test.shape[0])
    return test_loss

def linearRegression(X_train, Y_train, X_test, Y_test, alpha, iters):
    loss_his_train = []
    loss_his_test = []
    W = np.zeros((X_train.shape[1], 1))

    for i in range(iters):
        loss_train = computeCost(X_train, Y_train, W)
        loss_test = computeTestLoss(X_test, Y_test, W)
        loss_his_train.append(loss_train)
        loss_his_test.append(loss_test)

        W = gradientDescent(X_train, Y_train, W, alpha)

    return loss_his_train, loss_his_test

# 运行线性回归模型
alpha = 0.0001
iters = 10000
loss_his_train, loss_his_test = linearRegression(X_train_b, Y_train, X_test_b, Y_test, alpha, iters)

# 绘制训练和测试损失曲线
plt.figure(figsize=(6, 4))
plt.plot(np.arange(iters), loss_his_train, 'b', label='训练损失')
plt.plot(np.arange(iters), loss_his_test, 'r', label='测试损失')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('训练和测试损失随Epochs的变化')
plt.legend()
plt.show()
