import sys
sys.path.append("D:\python\lib\site-packages")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score, auc

import os
path = 'C:\\Users\\86153\\Desktop' + os.sep + "ex2data1.txt"#改成数据源路径
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])#手动指定header
data.head()
positive = data[data['Admitted'] == 1] # returns the subset of rows such Admitted = 1, i.e. the set of *positive* examples
negative = data[data['Admitted'] == 0] # returns the subset of rows such Admitted = 0, i.e. the set of *negative* examples

#描绘散点图
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

# 特征向量与标签提取
cols = data.shape[1]
data = data.values
x_data = data[:, 0:cols - 1]    # 去除标签列，留下特征向量
y_data = data[:, cols - 1:cols]     # 保留标签列


# 对数据进行归一化处理
def normalize(data):
    norm = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))    # 沿行方向（垂直方向）操作
    return norm


x_data = normalize(x_data)

# 建立逻辑回归模型 ， 使用sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# 损失函数
def computeCost(X, Y, W):
    P = sigmoid(np.dot(X, W))
    loss = np.sum(-Y * np.log(P) - (1 - Y) * np.log(1 - P)) / X.shape[0]
    return loss, P

# 计算损失函数对w参数的导数
def gradientDecent(W, X, Y):
    error = sigmoid(np.dot(X, W)) - Y
    grad = np.dot(X.T, error) / X.shape[1]
    W -= alpha * grad
    return W

# 逻辑回归参数训练过程
def logisticRegression(X, Y, iters):
    loss_his = []  # 初始化损失历史值
    W_his = []      # 初始化模型参数
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))  # 初始化W系数矩阵，w 是一个(feature_dim,1)矩阵
    # 重复步骤2和步骤3，直到收敛或迭代结束
    for i in range(iters):
        #  步骤2: 使用初始化的参数来预测输出值并计算出损失值
        loss, P = computeCost(X, Y, W)
        loss_his.append(loss)
        # 步骤3: 使用梯度下降法更新参数
        W_his.append(W.copy())
        W = gradientDecent(W, X, Y)
    return loss_his, W, W_his  # 返回损失和模型参数。


alpha = 0.0001
iters = 10000

# 初始化相关数据，用以存储每次的损失值以及查全率等数据
train_loss = []
test_loss = []

W_train_his = []
precision_his = []
recall_his = []
F1_score_his = []
fpr_his = []
tpr_his = []


# 5折交叉验证
kf = StratifiedKFold(n_splits=5, shuffle=True)

for train_index, test_index in kf.split(x_data, y_data):
    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    train_loss_his, W_train, W_his = logisticRegression(x_train, y_train, iters)
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

# 计算相关平均值
train_loss_avg = np.mean(train_loss, axis=0)
test_loss_avg = np.mean(test_loss, axis=0)
W_train_his_avg = np.mean(W_train_his, axis=0)

precision_avg = np.mean(precision_his)
recall_avg = np.mean(recall_his)
f1_avg = np.mean(F1_score_his)

fpr_grid = np.linspace(0.0, 1.0, 100)   # 初始化，0~1 的100数序列
mean_tpr = np.zeros_like(fpr_grid)
for i in range(5):
    mean_tpr += np.interp(fpr_grid, fpr_his[i], tpr_his[i])
fpr_grid = np.insert(fpr_grid, 0, 0.0)
mean_tpr = np.insert(mean_tpr, 0, 0.0)
mean_tpr = mean_tpr/5
roc_auc = auc(fpr_grid, mean_tpr)

# 打印所求数据
print("Precision： %f" % precision_avg)
print("Recall： %f" % recall_avg)
print("F1score： %f" % f1_avg)
print("Auc： %f" % roc_auc)

plt.plot(np.arange(iters), train_loss_avg, label='Training Loss')
plt.plot(np.arange(iters), test_loss_avg, label='Test Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curves')
plt.legend(['Train Loss', 'Test Loss'])
plt.show()

plt.plot(fpr_grid, mean_tpr, label='ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()