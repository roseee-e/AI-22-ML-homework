# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:52:26 2024

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from sklearn.linear_model import Ridge  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False 
}
rcParams.update(config)


path = r'C:\Users\lenovo\Downloads\regress_data1.csv'
import pandas as pd
data = pd.read_csv(path)
data.head()

cols = data.shape[1]
X_data = data.iloc[:,:cols-1]#X是所有行，去掉最后一列， 未标准化
y_data = data.iloc[:,cols-1:]#X是所有行，最后一列

# 原始数据图像
# data.plot(kind='scatter', x='人口', y='收益', figsize=(4,3)) # 利用散点图可视化数据
# import matplotlib
# plt.xlabel('人口')
# plt.ylabel('收益', rotation=90)
# plt.show()



# X_data.insert(0, 'Ones', 1)
# X_data.head()   #head()是观察前5行
# y_data.head()
# X=X_data.values
# Y=y_data.values
# W=np.array([[0.0],[0.0]])
# (X.shape,Y.shape, W.shape)


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
# alpha =0.0001
# iters = 10000
# loss_his, W = linearRegression(X,Y, alpha, iters)
# W
# x = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
# f = W[0, 0] + (W[1, 0] * x)

# fig, ax = plt.subplots(figsize=(6, 4))
# ax.plot(x, f, 'r', label='预测值')
# ax.scatter(X_data['人口'], data['收益'], label='训练数据')
# ax.legend(loc=2)
# ax.set_xlabel('人口' )
# ax.set_ylabel('收益', rotation=90)
# ax.set_title('预测收益和人口规模')
# plt.show()

# fig, ax = plt.subplots(figsize=(6, 4))
# ax.plot(np.arange(iters), loss_his, 'r')
# ax.set_xlabel('迭代次数')
# ax.set_ylabel('代价', rotation=0)
# ax.set_title('误差和训练Epoch数')
# plt.show()

#数据归一化
def normalize_minmax(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    X_norm = (X - min_val) / (max_val - min_val)
    return X_norm, min_val, max_val

X_data_norm, min_val, max_val = normalize_minmax(X_data.values)  # 对特征数据进行归一化处理

X_data_norm = np.insert(X_data_norm, 0, 1, axis=1)  # 在归一化后的数据前添加一列全1，代表x0
X = X_data_norm
Y = y_data.values
W = np.array([[0.0], [0.0]])  # 初始化W系数矩阵，w 是一个(2,1)矩阵

alpha = 0.01
iters = 5000
loss_his, W = linearRegression(X, Y, alpha, iters)
# print(W)

# 绘制图表
# x = np.linspace(0, 1, 100)  # 归一化后的人口数据范围是0到1
# f = W[0, 0] + (W[1, 0] * x)

# fig, ax = plt.subplots(figsize=(6, 4))

# ax.plot(x, f, 'r', label='预测值')
# ax.scatter(X[:,1], Y, label='训练数据')  # 注意这里使用归一化后的人口数据
# ax.legend(loc=2)
# ax.set_xlabel('人口')
# ax.set_ylabel('收益', rotation=90)
# ax.set_title('预测收益和人口规模')

#l2正则化
x_data = data.iloc[:, :-1]  # 特征  
y_data = data.iloc[:, -1]  # 目标变量  
  
# 划分训练集和测试集  
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)  
  
# 设置L2正则化系数
alpha = 0.6
  
# 创建Ridge回归模型实例并训练模型  
ridge_reg = Ridge(alpha=alpha)  
ridge_reg.fit(x_train, y_train)  
  

y_train_pred=ridge_reg.predict(x_train)

y_pred = ridge_reg.predict(x_test) 

plt.figure(figsize=(6,4)) 
plt.scatter(x_train.iloc[:, 0], y_train, label='训练数据')  
plt.plot(x_train.iloc[:, 0], y_train_pred, color='blue',  label='预测值')   
plt.xlabel('人口')  
plt.ylabel('收益')  
plt.title('l2正则化线性回归效果')  
plt.legend()  
plt.show()

#最小二乘法
x_data = data['人口'].values.reshape(-1, 1)  # 需要将一维数组转换为二维数组
y_data = data['收益'].values

poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(x_data)
model = LinearRegression()
model.fit(x_poly, y_data)

x_new = np.linspace(x_data.min(), x_data.max(), 100).reshape(-1, 1)
x_poly_new = poly_features.transform(x_new)
y_pred = model.predict(x_poly_new)

# 画图
plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.scatter(x_data, y_data, label='实际值')
plt.plot(x_new, y_pred, color='red', label='预测值')
plt.xlabel('人口')
plt.ylabel('收益')
plt.title('最小二乘法')
plt.legend()
plt.show()

#训练和测试损失曲线
# 划分训练集和测试集  
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)  
  
# 数据归一化  
scaler = StandardScaler()  # 创建归一化对象  
x_train_scaled = scaler.fit_transform(x_train)  # 对训练集进行归一化  
x_test_scaled = scaler.transform(x_test) 
W = np.zeros(x_train.shape[1])  
train_losses = []  
test_losses = []  
  
def computeCost(X, Y, W):  
    Y_hat = np.dot(X, W)  
    loss = np.sum((Y_hat - Y) ** 2) / (2 * X.shape[0])  
    return loss  
  
def gradientDescent(X, Y, W, alpha):  
    Y_hat = np.dot(X, W)  
    dW = (1 / X.shape[0]) * np.dot(X.T, (Y_hat - Y))  
    W -= alpha * dW  
    return W  
  
alpha = 0.0001  
iters=10000
def linearRegression(X, Y, W, alpha, iters):  
    loss_history = []  
    for i in range(iters):  
        W = gradientDescent(X, Y, W, alpha)  
        loss = computeCost(X, Y, W)  
        loss_history.append(loss)  
    return loss_history, W  
  
if __name__ == "__main__":  
    # 训练集上的损失和权重  
    train_losses, W_train = linearRegression(x_train_scaled, y_train, W, alpha, iters)  
      
    # 使用训练好的权重W_train在测试集上进行预测，并计算损失  
    test_losses = []  
    y_test_pred = np.dot(x_test_scaled, W_train)  
    test_loss = computeCost(x_test_scaled, y_test, W_train)  
    test_losses.append(test_loss)  # 将测试损失添加到列表中  
  
    # 假设我们想要画出每个epoch的损失，这里简单地将迭代次数作为epoch  
    epochs = list(range(iters))  
  
    plt.figure(figsize=(10, 6))  # 设置图表大小  
  
    # 绘制训练损失曲线  
    plt.plot(epochs, train_losses, label='训练损失', color='blue')  
    
    plt.plot(epochs, [test_loss] * len(epochs), label='测试损失', color='red', linestyle='--')  

    plt.title('训练和测试损失曲线')  
    plt.xlabel('Iteation')  
    plt.ylabel('Loss')  
  
    # 添加图例  
    plt.legend()  
  
    # 显示图表  
    plt.show()


