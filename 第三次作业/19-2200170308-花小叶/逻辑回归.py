# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:53:19 2024

@author: 86182
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score

  
# 配置matplotlib参数（保持不变）  
config = {  
    "mathtext.fontset": 'stix',  
    "font.family": 'serif',  
    "font.serif": ['SimHei'],  
    "font.size": 10,  
    'axes.unicode_minus': False  
}  
import matplotlib  
matplotlib.rcParams.update(config)  
  
# Sigmoid函数（保持不变）  
def sigmoid(z):  
    return 1 / (1 + np.exp(-z))  
  
# 计算逻辑回归损失函数（简化代码）  
def compute_cost(X, y, W):  
    P = sigmoid(np.dot(X, W))  
    loss = -np.sum(y * np.log(P) + (1 - y) * np.log(1 - P)) / len(y)  
    return loss, P  
  
# 梯度下降更新权重（简化代码）  
def gradient_descent(X, y, W, alpha):  
    error = sigmoid(np.dot(X, W)) - y  
    grad = np.dot(X.T, error) / len(y)  
    W -= alpha * grad  
    return W  
  
# 逻辑回归训练函数（使用向量化操作）  
def logistic_regression(X, y, alpha, iters):  
    loss_history = []  
    W_history = []  
    W = np.zeros(X.shape[1])  # 初始化权重为0  
    for _ in range(iters):  
        loss, _ = compute_cost(X, y, W)  
        loss_history.append(loss)  
        W_history.append(W.copy())  
        W = gradient_descent(X, y, W, alpha)  
    return loss_history, W_history, W  
  

path = r"E:\课程学习\机器学习\ex2data1.txt"
data = pd.read_csv(path, names=['Exam 1', 'Exam 2', 'Admitted'])
# print(data.head())
cols = data.shape[1]
X_data = data.iloc[:,:cols-1]#X是所有行，去掉最后一列， 未标准化
y_data = data.iloc[:,cols-1:]#X是所有行，最后一列
X_data=(X_data-X_data['Exam 1'].min())/(X_data['Exam 1'].max()-X_data['Exam 1'].min())
X_data=(X_data-X_data['Exam 2'].min())/(X_data['Exam 2'].max()-X_data['Exam 2'].min())

X_data.insert(0, 'Ones', 1)
X=X_data.values
y=y_data.values
# print(X_data)
alpha=0.1
iters=10000

train_losses_sum = []
test_losses_sum = []
W_train_his = []

precision_his = []
recall_his = []
f1_his = []
fpr_his = []
tpr_his = []
roc_auc_his = []

# 定义K折交叉验证对象
kf = KFold(n_splits=5)
# 使用K折交叉验证划分数据集
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    loss_his,W_his,W=logisticRegression(X_train,y_train,alpha,iters)

    test_loss=testRegression(X_test,y_test,W_his)

    test_losses_sum.append(test_loss)
    train_losses_sum.append(loss_his)
    W_train_his.append(W)

    P = sigmoid(np.dot(X_test, W))

    T = (P >= 0.5).astype(int)

    precision = precision_score(y_test, T)
    recall = recall_score(y_test, T)
    f1 = f1_score(y_test, T)
    fpr, tpr, _ = roc_curve(y_test, P)

    precision_his.append(precision)
    recall_his.append(recall)
    f1_his.append(f1)
    fpr_his.append(fpr)
    tpr_his.append(tpr)

avg_train_loss = np.mean(train_losses_sum,axis=0)
avg_test_loss = np.mean(test_losses_sum,axis=0)
W_train_his_avg = np.mean(W_train_his, axis=0)

# 计算评估指标的平均值
precision_avg = np.mean(precision_his)
recall_avg = np.mean(recall_his)
f1_avg = np.mean(f1_his)

fpr_grid = np.linspace(0.0, 1.0, 100)
mean_tpr = np.zeros_like(fpr_grid)
for i in range(5):
    mean_tpr += np.interp(fpr_grid, fpr_his[i], tpr_his[i])
fpr_grid = np.insert(fpr_grid, 0, 0.0)
mean_tpr = np.insert(mean_tpr, 0, 0.0)
mean_tpr = mean_tpr/5

plt.figure(figsize=(5, 3))
plt.plot(np.arange(iters), avg_train_loss, c='r', label='Training Loss')
plt.plot(np.arange(iters), avg_test_loss, c='b', label='Test Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curves')
plt.legend()
plt.show()

plt.figure(figsize=(5, 4))
plt.plot(fpr_grid, mean_tpr, color='red', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('平均ROC曲线')
plt.legend(loc='lower right')
plt.show()

roc_auc = auc(fpr_grid, mean_tpr)

print("Auc： %f\n" %roc_auc)
print("Precision： %f\n" % precision_avg)
print("Recall： %f\n" %recall_avg)
print("F1:  %f\n"% f1_avg)
