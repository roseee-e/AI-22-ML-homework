import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# 假设你已经有一个名为data.csv的数据集，其中包含特征和标签列
# 读取数据集
data = pd.read_csv('C:\\Users\\罗力彤\\Documents\\Tencent Files\\2827662737\\FileRecv\\ex2data1.txt')

# 假设数据集的最后一列是标签列，其余列是特征列
X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]   # 标签

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化逻辑回归模型
model = LogisticRegression()

# 训练模型
train_loss = []
val_loss = []
for epoch in range(100):  # 假设迭代100次
    model.fit(X_train, y_train)
    # 计算训练集和验证集的损失
    train_loss.append(model.score(X_train, y_train))
    val_loss.append(model.score(X_test, y_test))

# 绘制损失曲线
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 模型测试
y_pred = model.predict(X_test)

# 计算性能指标
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", auc)

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()