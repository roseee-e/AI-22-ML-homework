# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:24:01 2024

@author: lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# 读取txt文件并转换为DataFrame
# 读取数据
# data = pd.read_csv("D:\机器学习作业存放处\player_config_experiment.ini.txt")
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

path = r'D:\机器学习作业存放处\player_config_experiment.ini.txt'
data = pd.read_csv(path,header=None,names=['1','2','lab'])
lab0=data[data['lab']==0]
lab1=data[data['lab']==1]
x_data=data.iloc[:,:2]
y_data=data.iloc[:,2]
print(data)

#绘制散点图
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(lab0['1'],lab0['2'],s=30,c='black',marker='x',label='lab 0')
ax.scatter(lab1['1'],lab1['2'],s=30,c='r',marker='o',label='lab 1')
ax.legend()
ax.set_xlabel('1')
ax.set_ylabel('2')
ax.set_title('数据分布')
plt.show()

# 归一化函数
def normalize_minmax(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    X_norm = (X - min_val) / (max_val - min_val)
    return X_norm, min_val, max_val

X_data_norm, _, _ = normalize_minmax(x_data)  # 对特征数据进行归一化处理
X_data_norm = np.insert(X_data_norm, 0, 1, axis=1)  # 在归一化后的数据前添加一列全1，代表x0

# 定义逻辑回归模型函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义损失函数
def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

# 定义梯度下降函数
def gradient_descent(X, y, W, lr, iterations):
    m = len(y)
    losses = []
    for i in range(iterations):
        z = np.dot(X, W)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        W -= lr * gradient
        losses.append(loss(h, y))
    return W, losses

# 初始化权重
W = np.zeros(X_data_norm.shape[1])

# 设置学习率和迭代次数
lr = 0.01
iterations = 1000

# 使用梯度下降算法更新参数
W, losses = gradient_descent(X_data_norm, y_data, W, lr, iterations)
plt.figure()
plt.plot(losses)
plt.title('损失曲线')
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.show()
# 数据划分
X_train, X_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.2, random_state=42)

# 初始化逻辑回归模型
model = LogisticRegression()

# 模型训练
# train_losses = []
# val_losses = []
# num_iterations = 1000
# for i in range(num_iterations):
#     model.fit(X_train, y_train)
#     train_loss = -1 * np.mean(y_train * np.log(model.predict_proba(X_train)[:, 1]) + (1 - y_train) * np.log(1 - model.predict_proba(X_train)[:, 1]))
#     val_loss = -1 * np.mean(y_test * np.log(model.predict_proba(X_test)[:, 1]) + (1 - y_test) * np.log(1 - model.predict_proba(X_test)[:, 1]))
#     train_losses.append(train_loss)
#     val_losses.append(val_loss)

# # Plotting the training and validation loss curves
# plt.figure()
# plt.plot(range(1, num_iterations + 1), train_losses, label='Training Loss')
# plt.plot(range(1, num_iterations + 1), val_losses, label='Validation Loss')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# 训练模型
model = LogisticRegression()
model.fit(X_train, y_train)



# 预测
y_pred = model.predict(X_test)

# 模型评价
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#五折交叉验证法

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(model, x_data, y_data, cv=kfold, method='predict_proba')
fpr, tpr, thresholds = roc_curve(y_data, y_pred_cv[:,1])
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.show()

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
print("AUC: ", roc_auc)