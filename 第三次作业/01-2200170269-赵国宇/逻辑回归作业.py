import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置绘图的Matplotlib配置
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

# 读取数据
path = r'D:\python\ex2data1.csv'  ##使用原始字符串来表示文件路径
data = pd.read_csv(path)
X_data= data.iloc[:, :2]  ##选择前两列
y_data= data.iloc[:, 2]   ##选择第三列

#观察数据散点图
plt.figure(figsize=(8, 6))
plt.scatter(X_data[y_data==0].iloc[:, 0],X_data[y_data==0].iloc[:, 1], c='red', label='Label 0')# 标签为0的数据点
plt.scatter(X_data[y_data==1].iloc[:, 0],X_data[y_data==1].iloc[:, 1], c='blue',label='Label 1')# 标签为1的数据点
plt.title('Feature Value Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

#数据处理
def normalize_minmax(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    X_norm = (X - min_val) / (max_val - min_val)
    return X_norm, min_val, max_val
X_data_norm, min_val, max_val = normalize_minmax(X_data.values)  ##对特征数据进行归一化处理
X_data_norm = np.insert(X_data_norm, 0, 1, axis=1)  ##在归一化后的数据前添加一列全1，代表x0
X= X_data_norm
y = y_data.values.reshape(-1, 1)
feature_dim=X.shape[1] 
W=np.zeros((feature_dim,1))##初始化模型参数

#10次10折交叉验证——数据划分
from sklearn.model_selection import RepeatedKFold
kf = RepeatedKFold(n_splits=10,n_repeats=10,random_state=None)
for train_index,test_index in kf.split(X):
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]

#逻辑回归模型
def sigmoid(z):return 1/(1+np.exp(-z))##定义sigmoid函数
def computeCost(X, Y, W):
    P = sigmoid(np.dot(X, W))
    loss = np.sum(-Y * np.log(P) - (1 - Y) * np.log(1 - P)) / len(Y)
    return loss, P

def gradientDecent(W, X, Y, alpha):##定义梯度下降函数
    error = sigmoid(np.dot(X, W)) - Y
    grad = np.dot(X.T, error) / len(Y)
    W = W - alpha * grad
    return W

def logisticRegression(X, Y, alpha, iters):##定义模型函数
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

def testloss(X, Y, W_his, iters):##定义测试损失函数
    testloss_his = []
    for i in range(min(iters, len(W_his))): 
        loss, P = computeCost(X, Y, W_his[i])
        testloss_his.append(loss)
    return testloss_his

#设置超参数训练模型
alpha =0.0066
iters =100000
loss_his,  W_his,W = logisticRegression(X_train, y_train, alpha, iters)
testloss_his=testloss(X_test, y_test, W_his, iters)
print(f"W: '{W}'")

#绘制训练集和验证集的损失曲线
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(len(loss_his)), loss_his, 'r', label='Training Loss')
ax.plot(np.arange(len(testloss_his)), testloss_his, 'b', label='Test Loss')
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss', rotation=0)
ax.set_title('Training and Test Loss vs Iterations')
ax.legend()
plt.show()

# 模型评价
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score
p_train = sigmoid(np.dot(X, W))
pp = [(y_val[0], p_val[0]) for y_val, p_val in zip(y_train, p_train)]
##准备 ROC 曲线和 AUC 的数据
y_true = []
y_score = []
for p in pp:
    y_c = p[0]
    y = 1 if y_c == 1 else 0
    y_hat = p[1]
    y_true.append(y)
    y_score.append(y_hat)
##计算并打印精确度、召回率和 F1 分数
y_pred = [1 if prob > 0.5 else 0 for prob in y_score]
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
##计算 AUC并绘制 ROC 曲线
auc_score = roc_auc_score(y_true, y_score)
fpr, tpr, thresholds = roc_curve(y_true, y_score)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.title("ROC Curve", fontsize=14)
plt.ylabel('TPR', fontsize=14)
plt.xlabel('FPR', fontsize=14)
plt.legend(loc='lower right')
plt.show()