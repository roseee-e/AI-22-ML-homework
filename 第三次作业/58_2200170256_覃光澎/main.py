import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold

path = r"C:\Users\覃释天\Desktop\ex2data1.txt"
pdData = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])
pdData.head()

cols = pdData.shape[1]  # data.shape 返回一个元组，元组的第一个元素表示数据集的行数，第二个元素表示数据集的列数
X_data = pdData.iloc[:, :cols - 1]  # X去掉最后一列
Y_data = pdData.iloc[:, cols - 1:]  # Y 最后一列
x_data = X_data.values
y_data = Y_data.values

positive = pdData[pdData['Admitted'] == 1]
negative = pdData[pdData['Admitted'] == 0]
flg, ax = plt.subplots(figsize=(10, 5))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
ax.legend()
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def computeCost(X,Y,Z):
    p= sigmoid(np.dot(W.T,X))
    loss=np.sum(-Y*np.log(p)-(1-Y)*np.log(1-p))/X.shape[1]
    return loss,p
def gradientDecent(W,X,Y):
    error=sigmoid(np.dot(W.T,X))-Y
    grad=np.dot(X,error.T)/X.shape[1]
    W-=alpha*grad
    return W
# 逻辑回归参数训练过程
def logisticRegression(X, Y, iters, alpha):
    loss_his = []  # 初始化模型参数
    W_his = []
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))  # 初始化W系数矩阵，w 是一个(feature_dim,1)矩阵
    # 重复步骤2和步骤3，直到收敛或迭代结束
    for i in range(iters):
        # step2 : 使用初始化参数预测输出并计算损失
        loss, P = computeCost(X, Y, W)
        loss_his.append(loss)
        # step3: 采用梯度下降法更新参数
        W = gradientDecent(W, X, Y)
    return loss_his, W # 返回损失和模型参数。

#数据的归一化处理
def normalize_data(data):
    normalized_data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))  # axis=0沿着每列
    return normalized_data  # , np.min(data, axis=0), np.max(data, axis=0)
x_data = normalize_data(x_data)

alpha=0.001
iters=10000
loss_his, w=logisticRegression(X_data,Y_data,alpha,iters)

# 初始化相关数据，用以存储每次的损失值以及查全率等数据
train_loss = []
test_loss = []
W_train_his = []
precision_his = []
recall_his = []
F1_score_his = []
fpr_his = []
tpr_his = []

# k折交叉验证法
kf = StratifiedKFold(n_splits=5, shuffle=True)

for train_index, test_index in kf.split(x_data, y_data):
    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

train_loss_his, W_train, W_his = logisticRegression(x_train, y_train, alpha , iters)
W_train_his.append(W_train)

test_loss_his = []
for W in W_his:
    loss, _ = computeCost(x_test, y_test, W)
    test_loss_his.append(loss)

train_loss.append(train_loss_his)
test_loss.append(test_loss_his)

# 测试集预测结果
test_pred = sigmoid(np.dot(x_test, W_train))
binary_pred = (test_pred >= 0.5).astype(int)

precision_his.append(precision_score(y_test, binary_pred))
recall_his.append(recall_score(y_test, binary_pred))
F1_score_his.append(f1_score(y_test, binary_pred))
fpr, tpr, _ = roc_curve(y_test, test_pred)
fpr_his.append(fpr)
tpr_his.append(tpr)

avg_train_loss = np.mean(train_losses, axis=0)
avg_test_loss = np.mean(test_losses, axis=0)
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
# 画出训练和测试的损失曲线
plt.figure(figsize=(5, 3))
plt.plot(np.arange(iters), avg_train_loss, c='r', label='Training Loss')
plt.plot(np.arange(iters), avg_test_loss, c='b', label='Test Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curves')
plt.legend()
plt.show()



