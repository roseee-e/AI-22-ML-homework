import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams
from sklearn.model_selection import RepeatedKFold

import warnings
warnings.filterwarnings('ignore')

config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,  # 字号，大家自行调节
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)  # 设置画图的一些参数

# 读取数据
path = 'C:/Users/WXJ/OneDrive/Desktop/机器学习作业/regress_data2.csv'
data = pd.read_csv(path)
x_data = data.drop(columns=['价格'])
y_data = data['价格']

# 数据标准化
seta = pow(np.sum(x_data**2)/x_data.shape[0],0.5)
miu = np.sum(x_data)/x_data.shape[0]
x = (x_data-miu)/seta
y = y_data

x.insert(0, 'Ones', 1)

# 使用重复K折交叉验证数据划分法获得训练集和测试
kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=None)
for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]



X_train = x_train.values
Y_train = y_train.values.reshape(-1, 1)
X_test = x_test.values
Y_test = y_test.values.reshape(-1, 1)
def computeCost(X, Y, W, lamda):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y) ** 2) / (2 * num_train) + lamda * np.sum(W ** 2) / (2 * num_train)
    return loss


def gradientDescent(X, Y, W, alpha, lamda):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = (X.T @ (Y_hat - Y) + lamda * W) / num_train
    W -= alpha * dW
    return W


def linearRegression(X, Y, alpha, iters, lamda):
    loss_his = []
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss = computeCost(X, Y, W, lamda)
        loss_his.append(loss)
        W = gradientDescent(X, Y, W, alpha, lamda)
    return loss_his, W

alpha = 0.0006
lamda = 0.1
iters = 10000
loss_his, W = linearRegression(X_train, Y_train, alpha, iters, lamda)
print(W)

y_test_hat = []
loss_text = []
num_test = X_test.shape[0]
X_test_1=x_test.drop(columns=['Ones', '房间数']).values
X_test_2=x_test.drop(columns=['Ones', '面积']).values
y_test_hat = W[0,0] + (W[1,0] * X_test_1)+(W[2,0]*X_test_2)
for i in range(num_test):
    loss = ((y_test_hat[i] - Y_test[i]) ** 2) / (2 * num_test)
    loss_text.append(loss)

# 生成网格数据
x1_min, x1_max = x_train['面积'].min(), x_train['面积'].max()
x2_min, x2_max = x_train['房间数'].min(), x_train['房间数'].max()
x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
f = W[0, 0] + (W[1, 0] * x1) + (W[2,0]*x2)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')  # 创建一个三维的绘图工程
ax.set_title('价格与（面积和房间数）')  # 设置本图名称
ax.plot_surface(x1, x2, f)  # 绘制拟合平面

# 绘制测试数据点
ax.scatter(x_test['面积'], x_test['房间数'], y_test, c='r', label='测试数据')

ax.set_xlabel('面积')  # 设置x坐标轴
ax.set_ylabel('房间数')  # 设置y坐标轴
ax.set_zlabel('价格')  # 设置z坐标轴

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('误差和训练Epoch数')

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(num_test), loss_text, 'b')
ax.set_xlabel('测试集的个数序号')
ax.set_ylabel('代价', rotation=0)
ax.set_title('测试集的损失曲线')

plt.show()







