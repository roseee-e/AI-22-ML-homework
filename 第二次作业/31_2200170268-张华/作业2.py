import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams  # run command settings for plotting

config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)  ## 设置画图的一些参数
path = 'C:\\Users\\23052\Desktop\\regress_data1.csv'  # 注意斜杠的格式
data = pd.read_csv(path)

cols = data.shape[1]
X_data = data.iloc[:, :cols-1]
y_data = data.iloc[:, cols-1:]

X_data.insert(0, 'Ones', 1)

X = X_data.values
Y = y_data.values

def computeCost(X, Y, W):
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y)**2) / (2*X.shape[0])
    return loss

def gradientDescent(X, Y, W, alpha):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = X.T @ (Y_hat - Y) / X.shape[0]
    W += -alpha * dW
    return W

def linearRegression(X, Y, alpha, iters):
    loss_his = []
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss = computeCost(X, Y, W)
        loss_his.append(loss)
        W = gradientDescent(X, Y, W, alpha)
    return loss_his, W

alpha = 0.0001
iters = 10000
loss_train, W = linearRegression(X, Y, alpha, iters)

# Now let's calculate the test loss
X_test = X_data.values
Y_test = y_data.values
test_loss = computeCost(X_test, Y_test, W)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(len(loss_train)), loss_train, 'b', label='训练损失')
ax.axhline(y=test_loss, color='r', linestyle='--', label='测试损失')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('训练和测试损失曲线')
plt.legend()
plt.show()