import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams  ## run command settings for plotting
config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # �ֺţ�������е���
    'axes.unicode_minus': False # �����ţ���-��
}
rcParams.update(config)  ## ���û�ͼ��һЩ����
# ��ȡ����
path = 'C:\Users\cycy20\Downloads\regress_data1.csv'
data = pd.read_csv(path)
cols = data.shape[1]
X_data = data.iloc[:, :cols-1]
y_data = data.iloc[:, cols-1:]
# ���ݹ�һ��
X_data = (X_data - X_data.mean()) / X_data.std()
y_data = (y_data - y_data.mean()) / y_data.std()
X_data.insert(0, 'Ones', 1)
X = X_data.values
Y = y_data.values
W = np.array([[0.0], [0.0]])
def computeCostWithL2(X, Y, W, reg_lambda):
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y) ** 2) / (2 * X.shape[0]) + reg_lambda * np.sum(W[1:] ** 2) / 2
    return loss
def gradientDescentWithL2(X, Y, W, alpha, reg_lambda):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = (X.T @ (Y_hat - Y) + reg_lambda * np.vstack(([[0]], W[1:]))) / X.shape[0]
    W += -alpha * dW
    return W
def linearRegressionWithL2(X, Y, alpha, iters, reg_lambda):
    loss_his = []
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss = computeCostWithL2(X, Y, W, reg_lambda)
        loss_his.append(loss)
        W = gradientDescentWithL2(X, Y, W, alpha, reg_lambda)

    return loss_his, W
alpha = 0.0001
iters = 10000
reg_lambda = 0.1
loss_his, W = linearRegressionWithL2(X, Y, alpha, iters, reg_lambda)
def predict(X, W):
    y_pre = np.dot(X, W)
    return y_pre
# ����Ԥ������ѵ������ɢ��ͼ
x = np.linspace(X_data['�˿�'].min(), X_data['�˿�'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='Ԥ��ֵ')
ax.scatter(X_data['�˿�'], y_data, label='ѵ������')
ax.legend(loc=2)
ax.set_xlabel('�˿�')
ax.set_ylabel('����')
ax.set_title('Ԥ��������˿ڹ�ģ')

# ���ƴ��ۺ�������������仯������
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r')
ax.set_xlabel('��������')
ax.set_ylabel('����')
ax.set_title('����ѵ��Epoch��')
plt.show()