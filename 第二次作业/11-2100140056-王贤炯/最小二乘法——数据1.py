import numpy as np
import matplotlib.pyplot as plt
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
path = 'C:/Users/WXJ/OneDrive/Desktop/机器学习作业/regress_data1.csv'
import pandas as pd

data = pd.read_csv(path)
x_data = data.drop(columns=['收益'])
y_data = data['收益']

# 数据归一化
x = (x_data-x_data['人口'].min())/(x_data['人口'].max()-x_data['人口'].min())
y = data['收益']
x.insert(0, 'Ones', 1)

# 使用重复K折交叉验证数据划分法获得训练集和测试
kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=None)
for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# print(x_train)
# print(y_train)
X_train = x_train.values
Y_train = y_train.values.reshape(-1, 1)
X_test = x_test.values
Y_test = y_test.values.reshape(-1, 1)
def least_square_method(X, Y):
    # feature_dim = X.shape[0]
    # W = np.zeros((1,feature_dim))
    W = np.linalg.inv((X.T @ X)) @ X.T@Y
    return W


W = least_square_method(x_train, y_train)
# W[0]为bias
print(W)

y_test_hat = []
loss_text = []
num_test = X_test.shape[0]
X_test_1=x_test.drop(columns=['Ones']).values

y_test_hat = W[0] + (W[1] * X_test_1)
for i in range(num_test):
    loss = ((y_test_hat[i] - Y_test[i]) ** 2) / (2 * num_test)
    loss_text.append(loss)

x_ = np.linspace(x_train['人口'].min(), x_train['人口'].max(), 100)
f = W[0] + (W[1] * x_)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x_, f, 'r', label='预测值')
ax.scatter(x_test['人口'], y_test, label='测试数据')
ax.legend(loc=2)
ax.set_xlabel('人口')
ax.set_ylabel('收益', rotation=90)
ax.set_title('预测收益和人口规模')

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(num_test), loss_text, 'b')
ax.set_xlabel('测试集的个数序号')
ax.set_ylabel('代价', rotation=0)
ax.set_title('测试集的损失曲线')
plt.show()

plt.show()