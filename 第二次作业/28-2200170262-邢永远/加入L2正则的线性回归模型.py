import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import pandas as pd
from sklearn.model_selection import train_test_split
config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)


def computeCost_l(X, Y, W, l):
    Y_hat = np.dot(X,W)
    loss_l =(np.sum((Y_hat - Y)** 2)+np.sum(W** 2)*l)/(2*X.shape[0])# (m,n) @ (n, 1) -> (n, 1)
    return loss_l


def gradientDescent_l(X, Y, W, alpha, l):
    m = X.shape[0]
    Y_hat = np.dot(X,W)
    dW_l = (X.T@(Y_hat-Y)/ m) + l*W
#     dW = X.T@(Y_hat-Y)
    W += -alpha * dW_l
    return W

def linearRegression_l(X,Y, alpha, iters,l):
    loss_his = []
    # step1: initialize the model parameters
    feature_dim = X.shape[1]
    W=np.zeros((feature_dim,1)) ## 初始化W系数矩阵，w 是一个(feature_dim,1)矩阵
    ## repeat step 2 and step 3 untill to the convergence or the end of iterations
    for i in range (iters):
        # step2 : using the initilized parameters to predict the output and calculate the loss
        loss = computeCost_l(X,Y,W,l)
        loss_his.append(loss)
        # step3: using the gradient decent method to update the parameters
        W=gradientDescent_l(X, Y, W, alpha, l)
    return loss_his, W ## 返回损失和模型参数。

path = r"C:\Users\邢永远\OneDrive\桌面\regress_data1.csv"
data = pd.read_csv(path) ## data 是dataframe 的数据类型
data.head() # 返回data中的前几行数据，默认是前5行。

cols = data.shape[1]
X_data = data.iloc[:,:cols-1]#X是所有行，去掉最后一列， 未标准化
y_data = data.iloc[:,cols-1:]#X是所有行，最后一列
data.describe() ## 查看数据的统计信息

fig, (ax2,ax4) = plt.subplots(1, 2, figsize=(15, 4))  # 创建一个包含两个子图的图形



X_data.insert(0, 'Ones', 1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values
W0=np.array([[0.0],[0.0]])
W1=np.array([[0.0],[0.0]]) ## 初始化W系数矩阵，w 是一个(2,1)矩阵

alpha =0.0001
iters = 10000
l=0.1

loss_his1, W1 = linearRegression_l(X_train,y_train, alpha, iters,l)

x = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)

f1 = W1[0, 0] + (W1[1, 0] * x)



ax2.plot(x, f1, 'b', label='预测值')
ax2.scatter(X_data['人口'], y_data['收益'], label='训练数据')
ax2.legend(loc=2)
ax2.set_xlabel('人口')
ax2.set_ylabel('收益')
ax2.set_title('预测收益和人口规模')

ax4.plot(np.arange(iters), loss_his1, 'r')
ax4.set_xlabel('迭代次数')
ax4.set_ylabel('代价', rotation=0)
ax4.set_title('误差和训练Epoch数')
plt.show()
plt.tight_layout()  # 自动调整子图布局，以避免重叠

