
#1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置 matplotlib 中文显示
import matplotlib
from matplotlib import rcParams
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

# 读取数据
path = "C:\\Users\\Joe\Desktop\\regress_data2.csv"
data = pd.read_csv(path)
cols = data.shape[1]
X_data = data.iloc[:, :cols-1]
y_data = data.iloc[:, cols-1:]
X_data.insert(0, 'Ones', 1)
X = X_data.values
Y = y_data.values

# 计算线性回归
def computeCost(X, Y, W, lambda_val):
    Y_hat = np.dot(X, W)
    loss = (np.sum((Y_hat - Y) ** 2) + lambda_val * np.sum(W ** 2)) / (2 * X.shape[0])
    return loss

def gradientDescent(X, Y, W, alpha, lambda_val):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = (X.T @ (Y_hat - Y) + lambda_val * W) / num_train
    W += -alpha * dW
    return W

def linearRegression(X, Y, alpha, iters, lambda_val):
    loss_his = []
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss = computeCost(X, Y, W, lambda_val)
        loss_his.append(loss)
        W = gradientDescent(X, Y, W, alpha, lambda_val)
    return loss_his, W
# 获取训练得到的参数
alpha = 0.00000001
iters = 10000
lambda_val = 30000000 # 设置正则化参数
loss_his, W = linearRegression(X, Y, alpha, iters, lambda_val)
# 绘制原始数据点
plt.scatter(X[:, 1], Y, color='blue')
# 绘制拟合的线性回归直线
x_values = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
y_values = W[0] + W[1] * x_values
plt.plot(x_values, y_values, color='red', label='预测值')
plt.xlabel('面积')
plt.ylabel('价格')
plt.title(f'引入L2正则项且L2={lambda_val}',)
plt.legend()
plt.show()






#2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置 matplotlib 中文显示
import matplotlib
from matplotlib import rcParams
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

# 读取数据
path = "C:\\Users\\Joe\Desktop\\regress_data2.csv"
data = pd.read_csv(path)
cols = data.shape[1]
X_data = data.iloc[:, :cols-1]
y_data = data.iloc[:, cols-1:]
X_data.insert(0, 'Ones', 1)
X = X_data.values
Y = y_data.values

# 使用最小二乘法求解线性回归模型
W = np.linalg.inv(X.T @ X) @ X.T @ Y

# 输出参数矩阵 W
print("参数 W:")
print(W)

# 绘制原始数据点
plt.scatter(X[:, 1], Y, color='blue')

# 绘制拟合的线性回归直线
x_values = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
y_values = W[0] + W[1] * x_values
plt.plot(x_values, y_values, color='red', label='预测值')

plt.xlabel('面积')
plt.ylabel('价格')
plt.title('最小二乘法的面积与房价关系的预测')
plt.legend()
plt.show()




#3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置 matplotlib 中文显示
import matplotlib
from matplotlib import rcParams
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

# 读取数据
path = "C:\\Users\\Joe\Desktop\\regress_data2.csv"
data = pd.read_csv(path)
cols = data.shape[1]
X_data = data.iloc[:, :cols-1]
y_data = data.iloc[:, cols-1:]
X_data.insert(0, 'Ones', 1)
X = X_data.values
Y = y_data.values

# 分割训练集和测试集
def train_test_split(X, Y, test_size=0.3):
    num_total = X.shape[0]
    num_test = int(test_size * num_total)
    indices = np.random.permutation(num_total)
    train_indices = indices[num_test:]
    test_indices = indices[:num_test]
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    return X_train, X_test, Y_train, Y_test

# 计算线性回归
def computeCost(X, Y, W):
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y) ** 2) / (2 * X.shape[0])
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

# 获取训练得到的参数
alpha = 0.00000001
iters = 10000

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
loss_train, W_train = linearRegression(X_train, Y_train, alpha, iters)
loss_test = computeCost(X_test, Y_test, W_train)

# 绘制训练集和测试集的损失曲线
plt.plot(range(iters), loss_train, label='训练集')
plt.plot([0, iters], [loss_test, loss_test], 'r--', label='测试集')
plt.xlabel('训练次数')
plt.ylabel('损失')
plt.title('训练集和测试集的损失曲线')
plt.legend()
plt.show()

# 计算回归后的结果
Y_pred = np.dot(X, W_train)

# 绘制回归后的图像
plt.scatter(X[:, 1], Y, color='blue')
plt.plot(X[:, 1], Y_pred, color='red', label='预测值')
plt.xlabel('面积')
plt.ylabel('价格')
plt.title('回归后的面积与价格关系')
plt.legend()
plt.show()

# 绘制误差与训练epoch数的图像
plt.plot(range(iters), loss_train)
plt.xlabel('迭代次数')
plt.ylabel('代价')
plt.title('误差和训练Epoch数')
plt.show()




#4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# 设置 matplotlib 中文显示
import matplotlib
from matplotlib import rcParams
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

# 读取数据
data = pd.read_csv("C:\\Users\\Joe\Desktop\\regress_data2.csv")
X = data[['面积']].values
y = data['价格'].values

# 归一化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 多项式特征转换
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)

# 使用多项式回归拟合数据
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X_scaled, y)

# 绘制原始数据点
plt.scatter(X_scaled, y, color='blue')

# 绘制拟合曲线
X_plot = np.linspace(np.min(X_scaled), np.max(X_scaled), 100).reshape(-1, 1)
y_plot = model.predict(X_plot)
plt.plot(X_plot, y_plot, color='red', label='拟合曲线')

plt.xlabel('归一化后的面积')
plt.ylabel('价格')
plt.title('归一化后的回归拟合')
plt.legend()
plt.show()


