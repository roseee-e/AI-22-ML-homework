import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import rcParams
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
def computeCost(X, Y, W):
    b = 1e-5
    P = sigmoid(np.dot(W.T, X)) + b
    loss = np.sum(-Y * np.log(P) - (1 - Y) * np.log(1 - P)) / X.shape[1]
    return loss, P
def gradientDescent(W, X, Y, alpha):
    error = sigmoid(np.dot(W.T, X)) - Y
    grad = np.dot(X, error.T) / X.shape[1]
    W = W - alpha * grad
    return W
def logisticRegression(X, Y, X2, Y2, alpha, iters):
    loss_hist = []
    loss_hist2 = []
    feature_dim = X.shape[0]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss, p = computeCost(X, Y, W)
        loss_hist.append(loss)
        W = gradientDescent(W, X, Y, alpha)
        loss2, p2 = computeCost(X2, Y2, W)
        loss_hist2.append(loss2)
    return loss_hist, loss_hist2, W
# 从CSV文件加载数据
data = pd.read_csv('C:/Users/杨晒均/Documents/Tencent Files/2164528672/FileRecv/ex2data1.txt', header=None)
from sklearn.preprocessing import MinMaxScaler
# 获取原始数据
X_yuan = data.iloc[:, :2].values
y = data.iloc[:, 2].values.reshape(-1, 1)
# 创建MinMaxScaler实例并进行归一化处理
scaler = MinMaxScaler()
X = scaler.fit_transform(X_yuan)
# 输出归一化后的特征和标签
print(X)
print(y)
# 使用KFold进行交叉验证划分数据集
kf = KFold(n_splits=5)
train_losses = []
val_losses = []
# 交叉验证划分数据集
kf = KFold(n_splits=5)
train_losses = []
val_losses = []
# 参数设置
alpha = 0.003
iters = 10000
from sklearn.metrics import roc_auc_score, roc_curve
results = []
# 交叉验证并记录损失和指标
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index].T, X[val_index].T
    y_train, y_val = y[train_index].T, y[val_index].T
    
    train_loss, val_loss, W = logisticRegression(X_train, y_train, X_val, y_val, alpha, iters)  # 训练模型
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    _, y_pred = computeCost(X_val, y_val, W)
    
    # 计算 precision，recall，F1-score
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
# 绘制出损失曲线
plt.figure(figsize=(8, 6))
plt.plot(range(iters), np.mean(train_losses, axis=0), label='Training Loss')
plt.plot(range(iters), np.mean(val_losses, axis=0), label='Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Iterations',color='red')  
plt.legend()
# 计算平均指标
precision_avg = np.mean([result['precision'] for result in results])
recall_avg = np.mean([result['recall'] for result in results])
f1_avg = np.mean([result['f1_score'] for result in results])
auc_avg = np.mean([result['auc'] for result in results])
print("Average Precision:", precision_avg)
print("Average Recall:", recall_avg)
print("Average F1-score:", f1_avg)
print("Average AUC:", auc_avg)
# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
lw = 2
for result in results:
    plt.plot(result['fpr'], result['tpr'], lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC 曲线',color='red')   
plt.show()
