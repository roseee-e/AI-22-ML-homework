# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:46:42 2024

@author: 86182
"""

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt  
  
# 修改变量名以提高可读性  
file_path = "E:/课程学习/机器学习/regress_data1.csv"  
data = pd.read_csv(file_path)  
  
# 查看数据的前五行  
print(data.head())  
  
# 提取特征和目标变量  
x_data = data.iloc[:, :-1]  # 特征  
y_data = data.iloc[:, -1]  # 目标变量  
  
# 划分训练集和测试集  
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)  
  
# 数据归一化  
scaler = StandardScaler()  # 创建归一化对象  
x_train_scaled = scaler.fit_transform(x_train)  # 对训练集进行归一化  
x_test_scaled = scaler.transform(x_test)  # 对测试集进行归一化，使用训练集的均值和标准差  
  
# 初始化权重W和损失列表  
W = np.zeros(x_train.shape[1])  
train_losses = []  
test_losses = []  
  
def computeCost(X, Y, W):  
    Y_hat = np.dot(X, W)  
    loss = np.sum((Y_hat - Y) ** 2) / (2 * X.shape[0])  # 添加空格以提高可读性  
    return loss  
  
def gradientDescent(X, Y, W, alpha):  
    Y_hat = np.dot(X, W)  
    dW = (1 / X.shape[0]) * np.dot(X.T, (Y_hat - Y))  
    W -= alpha * dW  # 更新权重  
    return W  
  
alpha = 0.0001  
iters = 10000  
  
def linearRegression(X, Y, W, alpha, iters):  
    loss_history = []  
    for i in range(iters):  
        W = gradientDescent(X, Y, W, alpha)  
        loss = computeCost(X, Y, W)  
        loss_history.append(loss)  
    return loss_history, W  
  
if __name__ == "__main__":  
    # 训练集上的损失和权重  
    train_losses, W_train = linearRegression(x_train_scaled, y_train, W, alpha, iters)  
      
    # 使用训练好的权重W_train在测试集上进行预测，并计算损失  
    test_losses = []  
    y_test_pred = np.dot(x_test_scaled, W_train)  
    test_loss = computeCost(x_test_scaled, y_test, W_train)  
    test_losses.append(test_loss)  # 将测试损失添加到列表中  
  
    # 假设我们想要画出每个epoch的损失，这里简单地将迭代次数作为epoch  
    epochs = list(range(iters))  
  
    plt.figure(figsize=(10, 6))  # 设置图表大小  
  
    # 绘制训练损失曲线  
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')  
    
    plt.plot(epochs, [test_loss] * len(epochs), label='Test Loss', color='red', linestyle='--')  

    plt.title('Training and Test Loss over Iterations')  
    plt.xlabel('Iteration')  
    plt.ylabel('Loss')  
  
    # 添加图例  
    plt.legend()  
  
    # 显示图表  
    plt.show()