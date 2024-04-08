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

path = 'C:/Users/he  jiahao/source/repos/PythonApplication5/Python_test/regress_data1.csv'

import pandas as pd
data = pd.read_csv(path) 
head=data.head() 
cols = data.shape[1]
rows = data.shape[0]
X_data = data.iloc[:rows-30,:cols-1]#训练数据
Y_data = data.iloc[:rows-30,cols-1:]
X_test = data.iloc[rows-30:,:cols-1]#测试数据
Y_test = data.iloc[rows-30:,cols-1:]



data.plot(kind='scatter', x='人口', y='收益', figsize=(4,3)) 
plt.xlabel('人口')
plt.ylabel('收益', rotation=90)
plt.show()

X_data.insert(0, 'Ones', 1)
X=X_data.values
Y=Y_data.values
X_test.insert(0, 'Ones', 1)
X1=X_test.values
Y1=Y_test.values

W=np.array([[0.0],[0.0]]) 

def L2(W,your_data):
    l2=sum(W**2)*your_data/2
    return l2



def computeCost(X, Y, W,your_data):
    l2=L2(W,your_data)
    Y_hat = np.dot(X,W)
    loss1 =np.sum((Y_hat - Y)** 2)/(2*X.shape[0])
    loss2=np.sum(W** 2)/(2*X.shape[0])*l2
    loss=loss1+loss2
    return loss

def gradientDescent(X, Y, W, alpha,your_data): 
    l2=L2(W,your_data)
    Y_hat = np.dot(X,W)        
    dW = X.T@(Y_hat-Y)/ X.shape[0]
    W = W*(1- alpha*l2/X.shape[0])  -alpha * dW
    return W

def linearRegression(X,Y, alpha, iters,your_data):
    loss_his = []
    feature_dim = X.shape[1]
    W=np.zeros((feature_dim,1)) 
    for i in range (iters):
        loss = computeCost(X,Y,W,your_data)
        loss_his.append(loss)
        W=gradientDescent(X, Y, W, alpha,your_data)
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
loss_his_train, W = linearRegression(X,Y, alpha, iters,0.1)
loss_his_test = []
for i in range (iters):
        loss_test = computeCost(X1,Y1,W,0.1)
        loss_his_test.append(loss_test)





x = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)

print(W)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(X_data['人口'], Y_data['收益'], label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口' )
ax.set_ylabel('收益', rotation=90)
ax.set_title('梯度下降法引入L2的预测收益和人口规模')
plt.show()


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his_train, 'r',label='训练集损失函数')
ax.plot(np.arange(iters), loss_his_test, 'g',label='测试集损失函数')
ax.legend(loc=1)
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('梯度下降法引入L2的误差和训练Epoch数')
plt.show()

