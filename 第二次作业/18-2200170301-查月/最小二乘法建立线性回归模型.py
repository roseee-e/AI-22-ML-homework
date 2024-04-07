
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
  
# 设置matplotlib的字体参数  
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号  
  
# 读取CSV文件  
path = 'C:\\Users\\Lenovo\\Desktop\\regress_data1.csv'  
data = pd.read_csv(path)  
  
# 查看数据的前五行  
print(data.head())  
  
# 提取特征和目标变量  
cols = data.shape[1]  
X_data = data.iloc[:, :-1]  # 特征  
y_data = data.iloc[:, -1]  # 目标变量  
  
# 添加截距项到特征矩阵  
X_data['ones'] = 1  # 直接在DataFrame中添加一列  
X = np.hstack([X_data['ones'].values.reshape(-1, 1), X_data.iloc[:, :-1].values])  # 另一种方式添加截距项  
Y = y_data.values.reshape(-1, 1)  # 确保Y是二维数组  
  
# 使用最小二乘法计算权重矩阵W  
W = np.linalg.inv(X.T @ X) @ X.T @ Y  
  
# 使用权重矩阵进行预测  
Y_pred = X @ W  
  
# 绘制散点图和拟合线  
plt.scatter(X_data.iloc[:, -2], y_data, color='blue', label='数据点')  
plt.plot(X_data.iloc[:, -2], Y_pred[:, 0], color='red', label='拟合线', linewidth=2)    
plt.xlabel('人口')  
plt.ylabel('收益')  
plt.legend()  
plt.show()  
  
# 计算损失函数  
def computeCost(X, Y, W):  
    Y_hat = X @ W  
    loss = np.sum((Y_hat - Y) ** 2) / (2 * X.shape[0])  
    return loss  
  
cost = computeCost(X, Y, W)  
print(f'损失值: {cost}')