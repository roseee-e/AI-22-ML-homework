import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams  ## run command settings for plotting
import pandas as pd


l2=0.01
config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)

data = pd.read_csv(r"D:\python\data_input_study\regress_data1.csv")
cols = data.shape[1]##用与选取列数，2列
X_data = data.iloc[:,:cols-1]#X是所有行，去掉最后一列， 未标准化
y_data = data.iloc[:,cols-1:]#X是所有行，最后一列[9]
print(data.describe()) ## 查看数据的统计信息

data.plot(kind='scatter', x='人口', y='收益', figsize=(4,3)) # 利用散点图可视化数据
plt.xlabel('人口')
plt.ylabel('收益', rotation=90)
plt.show()

X_data.insert(0, 'Ones', 1)
X_data_first = X_data.iloc[:70]  # 前70个数据
X_data_last = X_data.iloc[70:]
y_data_first = y_data.iloc[:70]  # 前70个数据
y_data_last = y_data.iloc[70:]

X_train=X_data_first.values
Y_train=y_data_first.values
X_test=X_data_last.values
Y_test=y_data_last.values
W=np.array([[0.0],[0.0]])

def computeCost(X, Y, W,l2):
    Y_hat = np.dot(X,W)
    loss =np.sum((Y_hat - Y)**2)/(2*X.shape[0]) +l2*np.sum(W**2)/(2*X.shape[0])
    return loss

def gradientDescent(X, Y, W, alpha,l2):
    Y_hat = np.dot(X, W)
    dW = X.T @ (Y_hat - Y) / X.shape[0]
    W = W*(1- alpha*l2/X.shape[0]) - alpha*dW
    return W

def linearRegression(X,Y, alpha, iters,l2):
    loss_his = []
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss = computeCost(X, Y, W,l2)
        loss_his.append(loss)
        W = gradientDescent(X, Y, W, alpha,l2)
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
loss_his, W = linearRegression(X_train,Y_train, alpha, iters,l2)
loss_his_text = []
for i in range(iters):
    loss = computeCost(X_test, Y_test, W, l2)
    loss_his_text.append(loss)

x = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(X_data['人口'], data['收益'], label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口' )
ax.set_ylabel('收益', rotation=90)
ax.set_title('预测收益和人口规模，L2={0.01}')
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r',label='训练数据')
ax.plot(np.arange(iters), loss_his_text ,'g',label='测试数据')

ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('训练和测试损失曲线，L2={0.01}')
plt.legend(loc='upper right')
plt.show()







