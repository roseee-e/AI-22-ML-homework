import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# 加载数据
data = pd.read_csv(r"C:\Users\Lenovo\Documents\Tencent Files\3257956649\FileRecv\ex2data2.txt", header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 数据预处理
poly = PolynomialFeatures(degree=6)  # 添加多项式特征
X_poly = poly.fit_transform(X)
scaler = StandardScaler()  # 特征标准化
X_scaled = scaler.fit_transform(X_poly)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
def train_logistic_regression(X_train, y_train, X_val, y_val, C=1.0):
    model = LogisticRegression(C=C, max_iter=10000)
    train_losses, val_losses = [], []

    for i in range(100):  # 迭代100次
        model.fit(X_train, y_train)

        train_loss = -np.mean(y_train * np.log(model.predict_proba(X_train)[:, 1]) + (1 - y_train) * np.log(1 - model.predict_proba(X_train)[:, 1]))
        val_loss = -np.mean(y_val * np.log(model.predict_proba(X_val)[:, 1]) + (1 - y_val) * np.log(1 - model.predict_proba(X_val)[:, 1]))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    return model

# 训练模型并绘制损失曲线
model = train_logistic_regression(X_train, y_train, X_test, y_test)
# 模型测试
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 模型评价
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", auc)

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
