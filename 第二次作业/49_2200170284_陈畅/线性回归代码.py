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
    "font.size": 10,   # �ֺţ�������е���
    'axes.unicode_minus': False # �����ţ���-��
}
rcParams.update(config)  ## ���û�ͼ��һЩ����
## ��ȡ����
path = 'C:\Users\cycy20\Downloads\regress_data1.csv'
data = pd.read_csv(path) ## data ��dataframe ����������
data.head() # ����data�е�ǰ�������ݣ�Ĭ����ǰ5�С�
cols = data.shape[1]
X_data = data.iloc[:,:cols-1]#X�������У�ȥ�����һ�У� δ��׼��
y_data = data.iloc[:,cols-1:]#X�������У����һ��
X_data.insert(0, 'Ones', 1)
X_data.head()#head()�ǹ۲�ǰ5��
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
    W=np.zeros((feature_dim,1)) ## ��ʼ��Wϵ������w ��һ��(feature_dim,1)����
    for i in range (iters):
        loss = computeCost(X,Y,W)
        loss_his.append(loss)
        W=gradientDescent(X, Y, W, alpha)
    return loss_his, W ## ������ʧ��ģ�Ͳ�����
def predict(X, W):
    y_pre = np.dot(X,W)
    return y_pre
alpha =0.0001
iters = 10000
loss_his, W = linearRegression(X,Y, alpha, iters)
x = np.linspace(X_data['�˿�'].min(), X_data['�˿�'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='Ԥ��ֵ')
ax.scatter(X_data['�˿�'], data['����'], label='ѵ������')
ax.legend(loc=2)
ax.set_xlabel('�˿�' )
ax.set_ylabel('����', rotation=90)
ax.set_title('�ݶ��½�')
# ������ۺ���������L2������
def computeCostRegularized(X, Y, W, lamda):
    m = len(Y)
    error = np.dot(X, W) - Y
    cost = 1 / (2 * m) * np.sum(np.square(error)) + (lamda / (2 * m)) * np.sum(np.square(W[1:]))  # L2������
    return cost
# �ݶ��½�������L2������
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
# ������������ģ��
x = np.linspace(X_data['�˿�'].min(), X_data['�˿�'].max(), 100)
f_regularized = W[0] + (W[1] * x)
plt.figure(figsize=(6, 4))
plt.scatter(X_data['�˿�'], y_data, label='ѵ������')
plt.plot(x, f_regularized, 'r', label='����������')
plt.xlabel('�˿�')
plt.ylabel('����')
plt.legend()
plt.title('��������')
X_data = (X_data - X_data.mean()) / X_data.std()
y_data = (y_data - y_data.mean()) / y_data.std()
# ��ȡ�����ͱ�ǩ
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X = np.c_[np.ones(X.shape[0]), X]
W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
# Ԥ��
y_pred = X.dot(W)
# ���ƽ��
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 1], y, label='ѵ������')
plt.plot(X[:, 1], y_pred, color='red', label='Ԥ��ֵ')
plt.xlabel('�˿�')
plt.ylabel('����')
plt.legend()
plt.title('��С���˷�������Իع�ģ��')
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r')
ax.set_xlabel('��������')
ax.set_ylabel('����')
ax.set_title('����ѵ��Epoch��')
plt.show()


