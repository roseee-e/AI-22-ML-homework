import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

# 设置绘图参数
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def computeCost(X, y, W):
    b = 1e-5
    P = sigmoid(np.dot(W.T, X)) + b
    loss = np.sum(-y * np.log(P) - (1 - y) * np.log(1 - P)) / X.shape[1]
    return loss, P


def gradientDescent(W, X, y, alpha):
    error = sigmoid(np.dot(W.T, X)) - y
    grad = np.dot(X, error.T) / X.shape[1]
    W = W - alpha * grad
    return W


def logisticRegression(X_train, y_train, X_val, y_val, alpha, iters):
    loss_hist_train = []
    loss_hist_val = []
    feature_dim = X_train.shape[0]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss_train, p_train = computeCost(X_train, y_train, W)
        loss_hist_train.append(loss_train)
        W = gradientDescent(W, X_train, y_train, alpha)
        loss_val, p_val = computeCost(X_val, y_val, W)
        loss_hist_val.append(loss_val)
    return loss_hist_train, loss_hist_val, W


# 从指定文件加载数据
data = pd.read_csv(r'E:\机器学习\第三次实验\59-2200170292-黄家福\ex2data1.txt', header=None)

# 原始数据
X_yuan = data.iloc[:, :2].values
y = data.iloc[:, 2].values.reshape(-1, 1)

# 绘制原始数据的散点图，根据标签分类
plt.figure(figsize=(8, 6))
plt.scatter(X_yuan[:, 0], X_yuan[:, 1], c=y.ravel(), cmap='rainbow', edgecolors='k')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('原始数据散点图（按类别着色）')
plt.grid(True)
plt.show()

# 特征归一化
scaler = MinMaxScaler()
X = scaler.fit_transform(X_yuan)

# 交叉验证参数设定
kf = KFold(n_splits=5)
alpha = 0.003
iters = 10000

train_losses = []
val_losses = []
results = []

lw=2
# 交叉验证并记录损失和指标
all_fprs = []
all_tprs = []
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index].T, X[val_index].T
    y_train, y_val = y[train_index].T, y[val_index].T

    train_loss, val_loss, W = logisticRegression(X_train, y_train, X_val, y_val, alpha, iters)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    _, y_pred = computeCost(X_val, y_val, W)

     # 计算当前折叠的ROC曲线数据
    fpr, tpr, _ = roc_curve(y_val.squeeze(), y_pred.squeeze())
    all_fprs.append(fpr)
    all_tprs.append(tpr)

# 合并所有折叠的ROC曲线数据
mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []
for i in range(len(all_fprs)):
    interp_tpr = np.interp(mean_fpr, all_fprs[i], all_tprs[i])
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(auc(all_fprs[i], all_tprs[i]))

# 计算平均TPR及其标准差
mean_tpr = np.mean(tprs, axis=0)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

# 绘制综合ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(mean_fpr, mean_tpr, color='darkorange',
         lw=lw, label=f'Average ROC (AUC = {np.mean(aucs):.2f})')
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='black', alpha=.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('综合ROC曲线图', color='blue')
plt.legend(loc="lower right")
plt.show()

# 计算精度、召回率、F1分数
precision = precision_score(y_val.squeeze(), y_pred.squeeze() > 0.5)
recall = recall_score(y_val.squeeze(), y_pred.squeeze() > 0.5)
f1 = f1_score(y_val.squeeze(), y_pred.squeeze() > 0.5)

# 计算 AUC
fpr, tpr, _ = roc_curve(y_val.squeeze(), y_pred.squeeze())
auc_val = auc(fpr, tpr)

results.append({
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc_val,
        'fpr': fpr,
        'tpr': tpr
})

# 绘制损失曲线
plt.figure(figsize=(8, 6))
plt.plot(range(iters), np.mean(train_losses, axis=0), label='训练损失')
plt.plot(range(iters), np.mean(val_losses, axis=0), label='验证损失')
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('训练与验证损失随迭代次数变化图（数据已归一化，α=0.003）', color='blue')
plt.legend()
plt.show()
# 计算平均指标
precision_avg = np.mean([result['precision'] for result in results])
recall_avg = np.mean([result['recall'] for result in results])
f1_avg = np.mean([result['f1_score'] for result in results])
auc_avg = np.mean([result['auc'] for result in results])

print("平均Precision:", precision_avg)
print("平均Recall:", recall_avg)
print("平均F1-score:", f1_avg)
print("平均AUC:", auc_avg)
