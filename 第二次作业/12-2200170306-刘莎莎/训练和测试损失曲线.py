# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 23:06:45 2024

@author: 86182
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams  
config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   
    'axes.unicode_minus': False 
}
rcParams.update(config) 


## 读取数据
path = 'D:\\QQdocument\\regress_data1.csv'
import pandas as pd
data = pd.read_csv(path) 
#data.head() 
cols = data.shape[1]
X_data = data.iloc[:,:cols-1]
y_data = data.iloc[:,cols-1:]
data.plot(kind='scatter', x='人口', y='收益', figsize=(4,3))
import matplotlib
plt.xlabel('人口')
plt.ylabel('收益', rotation=90)
plt.show()
X_data.insert(0, 'Ones', 1)
X_data.head()
y_data.head()
X=X_data.values
Y=y_data.values
W=np.array([[0.0],[0.0]])

#看维度
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
#     dW = X.T@(Y_hat-Y)
    W += -alpha * dW
    return W
def linearRegression(X,Y, alpha, iters):
    loss_his = []
    feature_dim = X.shape[1]
    W=np.zeros((feature_dim,1)) 
    for i in range (iters):
        loss = computeCost(X,Y,W)
        loss_his.append(loss)
        W=gradientDescent(X, Y, W, alpha)
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
alpha =0.0001
iters = 10000
loss_his, W = linearRegression(X,Y, alpha, iters)
X_test = X_data.values
Y_test = y_data.values
test_loss = computeCost(X_test, Y_test, W)


#创建绘制图像
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(len(loss_his)), loss_his, 'b', label='训练损失')
ax.axhline(y=test_loss, color='r', linestyle='--', label='测试损失')

#设置坐标轴标签和标题
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('训练和测试损失曲线')
plt.legend()
plt.show()

# 



