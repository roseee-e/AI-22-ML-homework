import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, precision_score, recall_score, \
    f1_score

# 更新字体设置和图形大小
config = {
    "mathtext.fontset": 'stix',  # 数学公式的字体集
    "font.family": 'serif',  # 指定绘图中的字体系列为衬线字体
    "font.serif": ['SimHei'],  # 设置衬线字体的具体类型，中文宋体
    "font.size": 10,  # 字号，大家自行调节
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)  # 设置画图的一些参数
# 读取数据
path = r'D:\ex2data1 (1).txt'
data = pd.read_csv(path, header=None)  # 假设数据没有标题行
X_data = data.drop(2, axis=1)  # 使用drop方法选择前两列特征
y_data = data.iloc[:, 2]  # 仍然使用iloc选择第三列作为目标变量

# 观察数据散点图
plt.scatter(X_data[y_data == 0][0], X_data[y_data == 0][1], c='red', alpha=0.7, label='Label 0')
plt.scatter(X_data[y_data == 1][0], X_data[y_data == 1][1], c='blue', alpha=0.7, label='Label 1')
plt.title('特征值分布')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.legend()
plt.grid(True)  # 显示网格
plt.show()
# 数据处理 - 归一化函数
def scale_features_minmax(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    X_norm = (X - min_val) / (max_val - min_val)[:, np.newaxis]  # 使用np.newaxis确保形状匹配
    return X_norm, min_val, max_val


import numpy as np


# 数据归一化处理
def normalize_features(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    X_norm = (X - min_val) / (max_val - min_val)
    return X_norm, min_val, max_val


# 对特征数据进行归一化处理
X_norm, min_val, max_val = normalize_features(X_data)
# 在归一化后的数据前添加一列全1，代表x0，并重新赋值给X
X = np.insert(X_norm, 0, 1, axis=1)
y = y_data.values.reshape(-1, 1)

# 初始化模型参数
# 注意：为了改善模型的训练效果，可以使用随机初始化而不是全零初始化
np.random.seed(0)  # 设置随机种子以便结果可复现
feature_dim = X.shape[1]
W = np.random.randn(feature_dim, 1) * 0.01  # 使用小的随机数初始化W


# 逻辑回归模型
# 定义sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义损失函数（加入正则化项）
def computeCost(X, Y, W, reg_lambda):
    m = len(Y)
    P = sigmoid(np.dot(X, W))
    cost = -1 / m * (np.dot(Y.T, np.log(P)) + np.dot(1 - Y.T, np.log(1 - P)))
    cost += reg_lambda / (2 * m) * np.sum(np.square(W[:, 1:]))  # 正则化项（不包括偏置项）
    return cost

# 定义梯度下降函数
def gradientDecent(W, X, Y, alpha, reg_lambda):
    m = len(Y)
    error = sigmoid(np.dot(X, W)) - Y
    grad = (1 / m) * np.dot(X.T, error)
    grad[:, 1:] += (reg_lambda / m) * W[:, 1:]  # 对除了偏置项之外的参数添加正则化梯度
    W = W - alpha * grad
    return W

# 设置学习率和正则化参数
alpha = 0.01
reg_lambda = 0.01

# 定义模型函数
def logisticRegression(X, Y, alpha, iters):
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    loss_his = []
    W_his = []
    for i in range(iters):
        loss, P = computeCost(X, Y, W)
        loss_his.append(loss)
        W_his.append(W.copy())
        W = gradientDecent(W, X, Y, alpha)
    return loss_his, W_his, W


# 定义测试模型函数
def testmodel(X, Y, W_his, iters):
    testloss_his = []
    for i in range(min(iters, len(W_his))):
        loss, P = computeCost(X, Y, W_his[i])
        testloss_his.append(loss)
    return testloss_his, P


# 设置超参数训练模型
alpha = 0.0066
iters = 100000

# 存储变量
all_precision = 0
all_recall = 0
all_f1 = 0
all_auc = 0
fpr_list = []
tpr_list = []
loss_sum = []
testloss_sum = []
W_models = []

# 训练模型
kf = KFold(n_splits=5, shuffle=True, random_state=None)
# 5折交叉验证——数据划分
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # 训练集数据训练模型
    loss_his, W_his, W = logisticRegression(X_train, y_train, alpha, iters)
    # 测试集数据测试模型
    testloss_his, P = testmodel(X_test, y_test, W_his, iters)
    # 保存每一折的w
    W_models.append(W)
    # 累积测试损失和测试损失历史
    loss_sum.append(loss_his)
    testloss_sum.append(testloss_his)
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
testloss_average = np.mean(testloss_sum, axis=0)
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
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(len(loss_average)), loss_average, 'r', label='Training Loss')
ax.plot(np.arange(len(testloss_average)), testloss_average, 'b', label='Test Loss')
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss', rotation=0)
ax.set_title('Training and Test Loss vs Iterations')
ax.legend()
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
for i, W in enumerate(W_models, 1):
    print(f"Model {i}: W = {W.ravel()}")
print("Precision 平均值:", precision_average)
print("Recall 平均值:", recall_average)
print("F1 Score 平均值:", f1_average)