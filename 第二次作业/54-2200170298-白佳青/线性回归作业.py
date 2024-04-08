import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置绘图的Matplotlib配置
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)


path = 'D:/MachingLearning/regress_data1.csv'  # 使用原始字符串来表示文件路径
data = pd.read_csv(path)
cols = data.shape[1]
X_data = data.iloc[:,:cols-1]
y_data = data.iloc[:,cols-1:]

data.plot(kind='scatter', x='人口', y='收益', figsize=(4,3))

import matplotlib
plt.xlabel('人口')
plt.ylabel('收益', rotation=90)

plt.show()

def normalize_minmax(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    X_norm = (X - min_val) / (max_val - min_val)
    return X_norm, min_val, max_val

X_data_norm, min_val, max_val = normalize_minmax(X_data.values)  

X_data_norm = np.insert(X_data_norm, 0, 1, axis=1)
X = X_data_norm
Y = y_data.values
W = np.array([[0.0], [0.0]])


def computeCost(X, Y, W):
    Y_hat = np.dot(X,W)
    loss =np.sum((Y_hat - Y)** 2)/(2*X.shape[0])
    return loss

def gradientDescent(X, Y, W, alpha,lambda_):
    num_train = X.shape[0]     
    Y_hat = np.dot(X,W)        

    dW = X.T@(Y_hat-Y)/ X.shape[0]

    W =(1-alpha*lambda_/X.shape[0])*W-alpha * dW
    return W

def linearRegression(X,Y, alpha, iters,lambda_):
    loss_his = []

    feature_dim = X.shape[1]

    W=np.zeros((feature_dim,1))

    for i in range (iters):
       
        loss = computeCost(X,Y,W)
        loss_his.append(loss)
        
        W=gradientDescent(X, Y, W, alpha,lambda_)

    return loss_his, W

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

alpha = 0.01
lambda_ = 0.001
iters = 20000
loss_his, W = linearRegression(X, Y, alpha, iters, lambda_)
print(W)

# 绘制图表
x = np.linspace(0, 1, 100)
f = W[0, 0] + (W[1, 0] * x)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(x, f, 'r', label='预测值')
ax[0].scatter(X[:,1], Y, label='训练数据')
ax[0].legend(loc=2)
ax[0].set_xlabel('人口')
ax[0].set_ylabel('收益', rotation=90)
ax[0].set_title('预测收益和人口规模')

# 误差和训练Epoch数
ax[1].plot(np.arange(iters), loss_his, 'r')
ax[1].set_xlabel('迭代次数')
ax[1].set_ylabel('代价', rotation=0)
ax[1].set_title('误差和训练Epoch数')


# 调整子图间距
plt.tight_layout()
plt.show()

#最小二乘法求解线性回归模型
def least_squares(X, Y):
    W = np.linalg.inv(X.T @ X) @ X.T @ Y
    return W

W_least_squares = least_squares(X, Y)
print("最小二乘法求得的参数:\n", W_least_squares)

y_pred_train = predict(X, W_least_squares)


plt.figure(figsize=(6, 4))

plt.plot(X[:,1], y_pred_train, 'r', label='拟合曲线')
plt.scatter(X[:,1], Y, label='训练数据')
plt.legend()

plt.xlabel('人口')
plt.ylabel('收益', rotation=90)
plt.title('最小二乘法拟合训练数据')

plt.show()