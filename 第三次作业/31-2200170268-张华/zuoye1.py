
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# 加载数据集
data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, :2]  # 取前两个特征
y = data[:, 2]   # 目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化逻辑回归模型
logistic_model = LogisticRegression()

# 交叉验证
skf = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(logistic_model, X, y, cv=skf)

# 损失曲线
loss_curve = []
for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    logistic_model.fit(X_train_fold, y_train_fold)
    y_pred_fold = logistic_model.predict(X_test_fold)
    loss = -logistic_model.score(X_test_fold, y_test_fold)
    loss_curve.append(loss)

# 训练和绘制损失曲线
plt.plot(loss_curve)
plt.title('Cross Validation Loss Curve')
plt.xlabel('Folds')
plt.ylabel('Loss (Negative Score)')
plt.show()

# 训练逻辑回归模型
logistic_model.fit(X_train, y_train)

# 模型预测
y_pred = logistic_model.predict(X_test)

# 模型评价指标
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# 计算ROC曲线和AUC
y_pred_prob = logistic_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线和AUC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 打印分类报告
print(classification_report(y_test, y_pred))