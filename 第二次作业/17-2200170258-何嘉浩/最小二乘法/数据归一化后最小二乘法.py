import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams 
from sklearn.preprocessing import MinMaxScaler  

config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)  ## 设置画图的一些参数

path = 'C:/Users/he  jiahao/source/repos/PythonApplication5/Python_test/regress_data1.csv'

import pandas as pd
data = pd.read_csv(path) ## data 是dataframe 的数据类型
head=data.head() # 返回data中的前几行数据，默认是前5行。  
cols = data.shape[1]
X_data = data.iloc[:,:cols-1]#X是所有行，去掉最后一列， 未标准化
Y_data = data.iloc[:,cols-1:]#X是所有行，最后一列

data.plot(kind='scatter', x='人口', y='收益', figsize=(4,3)) # 利用散点图可视化数据
plt.xlabel('人口')
plt.ylabel('收益', rotation=90)
plt.show()


Y=Y_data.values
# 创建归一化对象  
scaler = MinMaxScaler()  
  
# 使用fit_transform方法对数据进行归一化  
X_normalized = scaler.fit_transform(X_data)


X_normalized = np.concatenate((np.ones((X_normalized.shape[0], 1)), X_normalized), axis=1)


W=np.array([[0.0],[0.0]]) ## 初始化W系数矩阵，w 是一个(2,1)矩阵

def computeCost(X, Y, W):
    Y_hat = np.dot(X,W)
    loss =np.sum((Y_hat - Y)** 2)/(2*X.shape[0])# (m,n) @ (n, 1) -> (n, 1)
    return loss

def coefficient(X,Y):
    coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y) 
    return coefficients


x = np.linspace(0, 1, 100)
coefficients=coefficient(X_normalized,Y)
f = coefficients[0,0]+(coefficients[1,0] * x)

Best_Cost=computeCost(X_normalized, Y, W)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(X_normalized[:,1], Y_data['收益'], label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口' )
ax.set_ylabel('收益', rotation=90)
ax.set_title('最小二乘法数据归一化后预测收益和人口规模')
plt.show()

print(Best_Cost)
