import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)
path = 'E:/qq/regress_data1.csv'
import pandas as pd
data = pd.read_csv(path) ## data 是dataframe 的数据类型
data.head() # 返回data中的前几行数据，默认是前5行。

cols = data.shape[1]
X_data = data.iloc[:,:cols-1]#X是所有行，去掉最后一列， 未标准化
y_data = data.iloc[:,cols-1:]#X是所有行，最后一列
print(X_data)
data.describe()

data.plot(kind='scatter', x='人口', y='收益', figsize=(4,3)) # 利用散点图可视化数据
import matplotlib
plt.xlabel('人口')
plt.ylabel('收益', rotation=90)
plt.show()

X_data.head()#head()是观察前5行
#print(X_data.head())
y_data.head()
# 假设 data 至少有 100 行
train_size = 70
test_size = data.shape[0] - train_size  # 确保剩余的行用于测试集

# 分割数据为训练集和测试集
X_train = X_data.iloc[:train_size]
y_train = y_data.iloc[:train_size]

X_test = X_data.iloc[train_size:]
y_test = y_data.iloc[train_size:]

X_data=X_train
y_data =y_train
def normalize_data(X_data):
    X_min = np.min(X_data)
    X_max = np.max(X_data)
    X_narm=(X_data-X_min)/(X_max-X_min)
    return X_narm
##print(X_data)
X_narm=normalize_data(X_data)
##Y_narm=normalize_data(y_data)
X_narm= np.insert(X_narm, 0, 1, axis=1)
X=X_narm
##print(X)
Y=y_data.values
W=np.array([[0.0],[0.0]])


def gradientDescent(X, Y, W, alpha, lambda_reg):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    # 计算梯度，包括L2正则化项
    dW = (1 / num_train) * np.dot(X.T, (Y_hat - Y)) + (lambda_reg / num_train) * W
    # 更新权重
    W -= alpha * dW
    return W


def computeCost(X, Y, W, lambda_reg):
    Y_hat = np.dot(X, W)
    # 计算损失，包括L2正则化项
    loss = (1 / (2 * X.shape[0])) * np.sum((Y_hat - Y) ** 2) + (lambda_reg / (2 * X.shape[0])) * np.sum(W ** 2)
    return loss

def linearRegression(X,Y, alpha, iters):
    loss_his = []
    # step1: initialize the model parameters
    feature_dim = X.shape[1]
    W=np.zeros((feature_dim,1)) ## 初始化W系数矩阵，w 是一个(feature_dim,1)矩阵
    ## repeat step 2 and step 3 untill to the convergence or the end of iterations
    lambda_reg = 0.01
    for i in range (iters):
        # step2 : using the initilized parameters to predict the output and calculate the loss

        loss = computeCost(X,Y,W,lambda_reg)
        loss_his.append(loss)
        # step3: using the gradient decent method to update the parameters
        W=gradientDescent(X, Y, W, alpha,lambda_reg)
    return loss_his, W ## 返回损失和模型参数。

def least_squares_fit(X, Y):
    W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return W

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
X_data=X_narm
alpha =0.001
iters = 50000
loss_his, W = linearRegression(X,Y, alpha, iters)

x = np.linspace(0, 1, 100)
W=least_squares_fit(X,Y)
f = W[0, 0] + (W[1, 0] * x)
y_pre= predict(X, W)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(X[:,1], Y, label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口' )
ax.set_ylabel('收益', rotation=90)
ax.set_title('预测收益和人口规模')
plt.show()


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r')
##ax.set_xlabel('迭代次数')
##ax.set_ylabel('代价', rotation=0)
##ax.set_title('误差和训练Epoch数')
##plt.show()

X_test_narm=normalize_data(X_test)
X_test_narm=np.insert(X_test_narm, 0, 1, axis=1)
X_test=X_test_narm
##print(X)
Y_test=y_test.values
W=np.array([[0.0],[0.0]])
loss_test_his, W = linearRegression(X_test,Y_test, alpha, iters)
##fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_test_his, 'g')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('误差和训练Epoch数')
plt.show()
