import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# 读取数据集
data = np.genfromtxt('D:\暂时文件\\ex2data1.txt', delimiter=',')
X = data[:, :-1]  # 特征
y = data[:, -1]   # 标签

# 数据归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 初始化超参数列表
C_values = np.logspace(-3, 3, 100)

# 初始化评价指标列表
precision_scores = []
recall_scores = []
f1_scores = []
auc_scores = []

# 初始化绘图
plt.figure(figsize=(10, 6))

# 五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化训练集和验证集的损失曲线
train_losses_avg = np.zeros(len(C_values))
val_losses_avg = np.zeros(len(C_values))

for i, (train_index, val_index) in enumerate(kf.split(X_scaled)):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 初始化模型
    clf = LogisticRegression()

    # 训练模型
    train_losses = []
    val_losses = []
    for C in C_values:
        clf.set_params(C=C, max_iter=1000, solver='sag', random_state=42)
        clf.fit(X_train, y_train)

        # 计算训练集和验证集上的损失
        train_loss = -np.mean(y_train * np.log(clf.predict_proba(X_train)[:, 1]) + (1 - y_train) * np.log(1 - clf.predict_proba(X_train)[:, 1]))
        val_loss = -np.mean(y_val * np.log(clf.predict_proba(X_val)[:, 1]) + (1 - y_val) * np.log(1 - clf.predict_proba(X_val)[:, 1]))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # 将每折的损失曲线加到平均损失曲线中
    train_losses_avg += np.array(train_losses)
    val_losses_avg += np.array(val_losses)

    # 测试模型
    y_pred = clf.predict(X_val)

    # 模型评价
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    auc_scores.append(auc)

# 绘制平均训练集和验证集的损失曲线
train_losses_avg /= kf.get_n_splits()
val_losses_avg /= kf.get_n_splits()
plt.semilogx(C_values, train_losses_avg, marker='o', label='Average Training Loss', linestyle='--')
plt.semilogx(C_values, val_losses_avg, marker='o', label='Average Validation Loss', linestyle='-')
plt.xlabel('C Values')
plt.ylabel('Loss')
plt.title('Average Training and Validation Loss')
plt.legend()

# 输出平均评价指标
print("Average Precision:", np.mean(precision_scores))
print("Average Recall:", np.mean(recall_scores))
print("Average F1 Score:", np.mean(f1_scores))
print("Average AUC:", np.mean(auc_scores))

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
for i, (train_index, val_index) in enumerate(kf.split(X_scaled)):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y[train_index], y[val_index]

    clf = LogisticRegression(C=1, max_iter=1000, solver='sag', random_state=42)
    clf.fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_val, clf.predict_proba(X_val)[:, 1])
    plt.plot(fpr, tpr, label=f'ROC Curve (Fold {i+1})', linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
