import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams  ## run command settings for plotting

config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)  ## 设置画图的一些参数
# 读取数据
path = 'C:/Users/14774/Downloads/regress_data1.csv'
data = pd.read_csv(path)
X_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1]

# 插入一列全为1的向量
X_data.insert(0, 'Ones', 1)

# 转换为numpy数组
X = X_data.values
y = y_data.values.reshape(-1, 1)

# 计算代价函数，加入L2正则项
def computeCostRegularized(X, y, W, lamda):
    m = len(y)
    error = np.dot(X, W) - y
    cost = 1 / (2 * m) * np.sum(np.square(error)) + (lamda / (2 * m)) * np.sum(np.square(W[1:]))  # L2正则项
    return cost

# 梯度下降，加入L2正则项
def gradientDescentRegularized(X, y, W, alpha, lamda, iters):
    m = len(y)
    cost_history = np.zeros(iters)
    
    for i in range(iters):        
        error = np.dot(X, W) - y
        W = W - (alpha / m) * (np.dot(X.T, error) + lamda * W)
        cost_history[i] = computeCostRegularized(X, y, W, lamda)
    
    return W, cost_history

alpha = 0.001
lamda = 0.1
iters = 10000
initial_W = np.zeros((X.shape[1], 1))
W, cost_history = gradientDescentRegularized(X, y, initial_W, alpha, lamda, iters)

# 引入正则项后的模型
x = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
f_regularized = W[0] + (W[1] * x)

plt.figure(figsize=(6, 4))
plt.scatter(X_data['人口'], y_data, label='训练数据')
plt.plot(x, f_regularized, 'r', label='引入正则项')
plt.xlabel('人口')
plt.ylabel('收益')
plt.legend()
plt.title('预测收益和人口规模-正则项系数为0.1-alpha=0.001',color='red')

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), cost_history, 'r')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('误差和训练Epoch数',color='red')
plt.show()
