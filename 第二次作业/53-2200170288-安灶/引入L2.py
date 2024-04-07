import sys
sys.path.append("D:\python\lib\site-packages")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams  ## run command settings for plotting
import pandas as pd
config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)  ## 设置画图的一些参数

## 读取数据
path = 'C:\\Users\\86178\\Documents\\Tencent Files\\3061593833\\FileRecv\\regress_data1.csv'

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
#plt.show()
X_data.insert(0, 'Ones', 1)
X_data.head()#head()是观察前5行
y_data.head()
X=X_data.values
Y=y_data.values
W=np.array([[0.0],[0.0]]) ## 初始化W系数矩阵，w 是一个(2,1)矩阵
(X.shape,Y.shape, W.shape)


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
reg_lambda = 0.1  # 正则化参数

loss_his, W = linearRegressionWithL2(X, Y, alpha, iters, reg_lambda)
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
loss_his, W = linearRegressionWithL2(X,Y, alpha, iters,reg_lambda)
x = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(X_data['人口'], data['收益'], label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口' )
ax.set_ylabel('收益', rotation=90)
ax.set_title('预测收益和人口规模(引入L2)')
plt.show()
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('误差和训练Epoch数(引入L2)')
plt.show()
