import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams 
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
data.head()

cols = data.shape[1]
X_data = data.iloc[:, :cols - 1]
y_data = data.iloc[:, cols - 1:]

X_data.insert(0, 'Ones', 1)

X = X_data.values
Y = y_data.values

def computeCost(X, Y, W, lambda_):
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y) ** 2) / (2 * X.shape[0])
    regularization_term = lambda_ * np.sum(W ** 2)
    total_loss = loss + regularization_term
    return total_loss

def gradientDescent(X, Y, W, alpha, lambda_):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = X.T @ (Y_hat - Y) / X.shape[0] + lambda_ * W
    W += -alpha * dW
    return W

def linearRegression(X, Y, alpha, iters, lambda_):
    loss_his = []
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss = computeCost(X, Y, W, lambda_)
        loss_his.append(loss)
        W = gradientDescent(X, Y, W, alpha, lambda_)
    return loss_his, W

def predict(X, W):
    y_pre = np.dot(X, W)
    return y_pre
alpha = 0.01
iters = 1000
lambda_ = 0.5
loss_his, W = linearRegression(X, Y, alpha, iters, lambda_)
x = np.linspace(X_data['�˿�'].min(), X_data['�˿�'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)
# ���ɲ�������
np.random.seed(0)
test_size = 20
X_test = np.random.rand(test_size, 2)
X_test[:, 0] = 1  # Set the bias term for the test data
Y_test = np.dot(X_test, np.array([[3], [2]]))  # Some arbitrary coefficients for testing
# Ԥ����Լ����
Y_test_pred = predict(X_test, W)
test_loss = computeCost(X_test, Y_test, W, lambda_)
# ����ѵ���Ͳ�����ʧ����
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r', label='ѵ����')
ax.axhline(y=test_loss, color='b', linestyle='-', label='���Լ�')
ax.set_xlabel('��������')
ax.set_ylabel('����', rotation=0)
ax.set_title('ѵ���Ͳ�����ʧ')
ax.legend()
plt.show()
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='Ԥ��ֵ')
ax.scatter(X_data['�˿�'], data['����'], label='ѵ������')
ax.legend(loc=2)
ax.set_xlabel('�˿�')
ax.set_ylabel('����', rotation=90)
ax.set_title('Ԥ��������˿ڹ�ģ')
plt.show()