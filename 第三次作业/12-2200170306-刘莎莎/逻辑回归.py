# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:28:26 2024

@author: 86182
"""






from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 加载数据
data = pd.read_csv("D:\\QQdocument\\ex2data1.txt", sep=',', names=['f1', 'f2', 'admitted'])
x = data.iloc[:, :2]
y = data.iloc[:, 2]

#数据分布

plt.figure(figsize=(10, 5))
plt.scatter(x[y==0].iloc[:, 0],x[y==0].iloc[:, 1], c='red', label='Admitted')
plt.scatter(x[y==1].iloc[:, 0],x[y==1].iloc[:, 1], c='blue',label='not Admitted')

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()



# 归一化函数
def normalize_minmax(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    X_norm = (X - min_val) / (max_val - min_val)
    return X_norm, min_val, max_val

# 归一化特征数据
x_data_norm, min_val, max_val = normalize_minmax(x.values)
# 在归一化后的数据前添加一列全1，代表x0，并重新赋值给x
x_data_norm = np.insert(x_data_norm, 0, 1, axis=1)
x = x_data_norm
y = y.values.reshape(-1, 1)
# print(x.shape)

# 自定义逻辑回归类
class LogisticRegression:
    def __init__(self, alpha=0.0001, iters=5000):
        self.alpha = alpha
        self.iters = iters
        self.loss_his = []
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def computeCost(self, X, y, theta):
        m = X.shape[0]
        p = self.sigmoid(np.dot(X, theta))
        # 使用mean()替代sum()除以m计算平均损失
        cost = (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean()
        return cost

    def gradientDescent(self, X, y, theta):
        m = X.shape[0]
        error = self.sigmoid(np.dot(X, theta)) - y
        grad = (1 / m) * np.dot(X.T, error)
        return theta - self.alpha * grad

    def fit(self, X, y):
        #feature_dim = X.shape[1]
        feature_dim = len(x[1])
        self.theta = np.zeros((feature_dim, 1))
        for i in range(self.iters):
            cost = self.computeCost(X, y, self.theta)
            self.loss_his.append(cost)
            self.theta = self.gradientDescent(X, y, self.theta)

    def predict(self, X):
        """
        预测新数据点的结果
        """
        p = self.sigmoid(np.dot(X, self.theta))
        return (p >= 0.5).astype(int).reshape(-1, 1)


alpha = 0.0001
iters = 5000
lr = LogisticRegression(alpha, iters)

# 5次5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)


val_losses = []
test_losses = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []
tprs = []
fprs = []

for i in range(5):  # 5次交叉验证
    val_loss_per_iter = []
    test_loss_per_iter = []
    for train_index, (val_index, test_index) in enumerate(kf.split(x)):
        X_train, X_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # 拟合模型
        lr.fit(X_train, y_train)
        
        # 计算验证集损失
        val_loss_per_iter.append(lr.computeCost(X_val, y_val, lr.theta))
        
        
        X_test = x[test_index]
        y_test = y[test_index]
        test_loss_per_iter.append(lr.computeCost(X_test, y_test, lr.theta))
        
        # 预测概率
        probs = lr.predict_proba(X_val)[:, 1]  # 假设正类标签是1
        
        # 计算精度、召回率和F1分数
        y_val_pred = (probs > 0.5).astype(int)  # 根据概率阈值生成预测标签
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_val, y_val_pred, average='binary')
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
        # 计算ROC曲线和AUC
        fpr, tpr, thresholds = roc_curve(y_val, probs)
        roc_auc = auc(fpr, tpr)
        roc_auc_scores.append(roc_auc)
        tprs.append(tpr)
        fprs.append(fpr)
    
    # 存储每次交叉验证的平均验证集和测试集损失
    val_losses.append(np.mean(val_loss_per_iter, axis=0))
    test_losses.append(np.mean(test_loss_per_iter, axis=0))


plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_scores[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")



# 绘制损失曲线
plt.figure(figsize=(10, 5))

# 绘制验证集损失曲线
for i, val_loss in enumerate(val_losses):
    plt.plot(val_loss, label=f'Validation Loss Fold {i+1}')

# 绘制测试集损失曲线
for i, test_loss in enumerate(test_losses):
    plt.plot(test_loss, linestyle='--', label=f'Test Loss Fold {i+1}')

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Validation and Test Loss Over Iterations')
plt.legend()
plt.show()





























