import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, log_loss
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv( "C:\\Users\\小黑\\Desktop\\机器学习\\第三次\\26_2200170246_莫永清\\ex2data1.txt", header=None)

# 数据预处理
X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values   # 标签

# 数据划分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化模型
model = LogisticRegression(solver='liblinear', max_iter=1, warm_start=True)

# 绘制数据散点图
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Class 0')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Class 1')
plt.title('Training Data Scatter Plot')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# 训练模型并记录损失
train_losses = []
val_losses = []
for _ in range(10):
    model.fit(X_train, y_train)
    y_train_pred_prob = model.predict_proba(X_train)
    y_val_pred_prob = model.predict_proba(X_val)
    train_losses.append(log_loss(y_train, y_train_pred_prob))
    val_losses.append(log_loss(y_val, y_val_pred_prob))

# 绘制损失曲线
plt.subplot(1, 3, 2)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training vs Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.legend()

# 模型评估及绘制ROC曲线
precision = precision_score(y_val, model.predict(X_val))
recall = recall_score(y_val, model.predict(X_val))
f1 = f1_score(y_val, model.predict(X_val))
auc_score = roc_auc_score(y_val, y_val_pred_prob[:, 1])
fpr, tpr, _ = roc_curve(y_val, y_val_pred_prob[:, 1])

plt.subplot(1, 3, 3)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()

plt.tight_layout()
plt.show()

# 输出模型性能指标
print("精度: {:.3f}".format(precision))
print("召回率: {:.3f}".format(recall))
print("F1 分数: {:.3f}".format(f1))
print("AUC 分数: {:.3f}".format(auc_score))