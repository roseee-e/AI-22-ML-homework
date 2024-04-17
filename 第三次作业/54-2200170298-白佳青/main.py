import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, roc_curve, auc, precision_score, recall_score, f1_score

# 读取数据
path = r'D:\Pycharm-project\pythonProject\data\ex2data1.txt'  # 使用原始字符串来表示文件路径
data = pd.read_csv(path)
X_data = data.iloc[:, :2]  # 选择前两列
y_data = data.iloc[:, 2]  # 选择第三列

# 观察数据散点图
plt.figure(figsize=(8, 6))
plt.scatter(X_data[y_data == 0].iloc[:, 0], X_data[y_data == 0].iloc[:, 1], c='red', label='Label 0')
plt.scatter(X_data[y_data == 1].iloc[:, 0], X_data[y_data == 1].iloc[:, 1], c='blue', label='Label 1')
plt.title('Feature Value Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 数据处理
# 定义归一化函数
def normalize_minmax(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    X_norm = (X - min_val) / (max_val - min_val)
    return X_norm, min_val, max_val

# 对特征数据进行归一化处理
X_data_norm, _, _ = normalize_minmax(X_data.values)
# 在归一化后的数据前添加一列全1，代表x0，并重新命名
X_data_norm = np.insert(X_data_norm, 0, 1, axis=1)
X = X_data_norm
y = y_data.values.reshape(-1, 1)

# 逻辑回归模型
# 定义sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义损失函数
def computeCost(X, Y, W):
    P = sigmoid(np.dot(X, W))
    loss = np.sum(-Y * np.log(P) - (1 - Y) * np.log(1 - P)) / len(Y)
    return loss, P

# 定义梯度下降函数
def gradientDecent(W, X, Y, alpha):
    error = sigmoid(np.dot(X, W)) - Y
    grad = np.dot(X.T, error) / len(Y)
    W -= alpha * grad
    return W

# 定义训练模型函数
def train_model(X, Y, alpha, iters):
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    loss_his = []
    for i in range(iters):
        loss, _ = computeCost(X, Y, W)
        loss_his.append(loss)
        W = gradientDecent(W, X, Y, alpha)
    return loss_his, W

# 设置超参数训练模型
alpha = 0.01
iters = 60000

# 存储变量
all_precision = 0
all_recall = 0
all_f1 = 0
all_auc = 0
fpr_list = []
tpr_list = []
loss_sum = []

# 训练模型
kf = KFold(n_splits=5, shuffle=True, random_state=None)
# 5折交叉验证——数据划分
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # 训练模型
    loss_his, W = train_model(X_train, y_train, alpha, iters)
    loss_sum.append(loss_his)
    # 测试模型
    _, P = computeCost(X_test, y_test, W)
    # 计算 Precision、Recall、F1 Score
    precision = precision_score(y_test, np.round(P))
    recall = recall_score(y_test, np.round(P))
    f1 = f1_score(y_test, np.round(P))
    all_precision += precision
    all_recall += recall
    all_f1 += f1
    # 计算fpr，tpr，auc
    fpr, tpr, _ = roc_curve(y_test, P)
    roc_auc = auc(fpr, tpr)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    all_auc += roc_auc

# 计算平均损失
loss_average = np.mean(loss_sum, axis=0)
# 计算平均Precision、Recall、F1 Score
precision_average = all_precision / 5
recall_average = all_recall / 5
f1_average = all_f1 / 5
# 计算平均fpr，tpr，auc
max_len = max(len(fpr) for fpr in fpr_list)
extended_fpr_list = [np.concatenate([fpr, np.full(max_len - len(fpr), np.nan)]) for fpr in fpr_list]
extended_tpr_list = [np.concatenate([tpr, np.full(max_len - len(tpr), np.nan)]) for tpr in tpr_list]
fpr_average = np.nanmean(extended_fpr_list, axis=0)
tpr_average = np.nanmean(extended_tpr_list, axis=0)
auc_average = all_auc / 5

# 绘制训练集和验证集的损失曲线
plt.figure(figsize=(8, 4))
plt.plot(np.arange(len(loss_average)), loss_average, 'r', label='Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss vs Iterations')
plt.legend()
plt.show()

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr_average, tpr_average, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_average)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 打印相关数据
print("Precision 平均值:", precision_average)
print("Recall 平均值:", recall_average)
print("F1-Score 平均值:", f1_average)