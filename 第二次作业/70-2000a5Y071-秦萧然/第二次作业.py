import matplotlib
matplotlib.use('Qt5Agg')
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import RepeatedKFold
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
path = 'D:\\视频\\机器学习作业\\regress_data1.csv'
import pandas as pd

data = pd.read_csv(path)
x_data = data.drop(columns=['收益'])
y_data = data['收益']

# 数据归一化
x = (x_data - x_data['人口'].min()) / (x_data['人口'].max() - x_data['人口'].min())
y = data['收益']

x.insert(0, 'Ones', 1)
# print(x)

## x的特征维度为1，不需要做数据的标准化或归一化

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


alpha = 0.01
lamda = 1
iters = 10000
loss_his, W = linearRegression(X_train, Y_train, alpha, iters, lamda)

y_test_hat = []
loss_text = []
num_test = X_test.shape[0]
X_test_1 = x_test.drop(columns=['Ones']).values

y_test_hat = W[0, 0] + (W[1, 0] * X_test_1)
for i in range(num_test):
    loss = ((y_test_hat[i] - Y_test[i]) ** 2) / (2 * num_test)
    loss_text.append(loss)

x = np.linspace(x_train['人口'].min(), x_train['人口'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(x_test['人口'], y_test, label='测试数据')
ax.legend(loc=2)
ax.set_xlabel('人口')
ax.set_ylabel('收益', rotation=90)
ax.set_title('预测收益和人口规模')

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
