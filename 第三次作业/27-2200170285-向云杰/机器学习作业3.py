import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score

config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

# 读取数据
path = r"C:\Users\云\Desktop\ex2data1.txt"  
data = pd.read_csv(path)
X_data= data.iloc[:, :2]  
y_data= data.iloc[:, 2]   

#数据处理
def normalize_minmax(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    X_norm = (X - min_val) / (max_val - min_val)
    return X_norm, min_val, max_val

X_data_norm, min_val, max_val = normalize_minmax(X_data.values)
X_data_norm = np.insert(X_data_norm, 0, 1, axis=1) 
X= X_data_norm
y = y_data.values.reshape(-1, 1)
alpha = 0.0066
iters = 100000

#定义逻辑回归模型
def sigmoid(z):
    return 1/(1+np.exp(-z))

def computeCost(X, Y, W):
    P = sigmoid(np.dot(X, W))
    loss = np.sum(-Y * np.log(P) - (1 - Y) * np.log(1 - P)) / len(Y)
    return loss, P

def gradientDecent(W, X, Y, alpha):
    error = sigmoid(np.dot(X, W)) - Y
    grad = np.dot(X.T, error) / len(Y)
    W = W - alpha * grad
    return W

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

def testmodel(X, Y, W_his, iters):
    testloss_his = []
    for i in range(min(iters, len(W_his))): 
        loss, P = computeCost(X, Y, W_his[i])
        testloss_his.append(loss)
    return testloss_his, P

#记录结果
all_precision = 0
all_recall = 0
all_f1 = 0
all_auc = 0
fpr_list = []
tpr_list = []
loss_sum = []
testloss_sum = []
W_models = []

#5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=None)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    loss_his,W_his, W = logisticRegression(X_train, y_train, alpha, iters)
    testloss_his, P = testmodel(X_test, y_test, W_his, iters)

    loss_sum.append(loss_his)  
    testloss_sum.append(testloss_his) 
    W_models.append(W)

    precision = precision_score(y_test, np.round(P))
    recall = recall_score(y_test, np.round(P))
    f1 = f1_score(y_test, np.round(P))
    all_precision+=precision
    all_recall+=recall
    all_f1+=f1

    fpr, tpr, _ = roc_curve(y_test, P)
    roc_auc = auc(fpr, tpr)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    all_auc+=roc_auc

loss_average = np.mean(loss_sum, axis=0)
testloss_average = np.mean(testloss_sum, axis=0)

precision_average=all_precision/5
recall_average=all_recall/5
f1_average=all_f1/5
auc_average=all_auc/5

#绘制损失曲线
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(len(loss_average)), loss_average, 'r', label='Training Loss')
ax.plot(np.arange(len(testloss_average)), testloss_average, 'b', label='Test Loss')
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss', rotation=0)
ax.legend()
plt.show()


# 计算平均的fpr和tpr
fpr_all = np.unique(np.concatenate(fpr_list))
mean_tpr = np.zeros_like(fpr_all)
for i in range(5):
    mean_tpr += np.interp(fpr_all, fpr_list[i], tpr_list[i])
mean_tpr /= 5

# 绘制ROC并计算AUC
roc_auc = auc(fpr_all, mean_tpr)
plt.figure()
lw = 2
plt.plot(fpr_all, mean_tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 打印模型参数以及评估结果
for i, W in enumerate(W_models, 1):
    print(f"Model {i}: W = {W.ravel()}")
print("Precision 平均值:", precision_average)
print("Recall 平均值:", recall_average)
print("F1 Score 平均值:", f1_average)
print("AUC 平均值:", auc_average)
