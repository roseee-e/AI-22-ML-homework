# 这是一个示例 Python 脚本。
# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams  ## run command settings for plotting
config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)  ## 设置画图的一些参数

## 读取数据
# 标题行内容
path = 'E:/软件/QQ/ex2data1.txt'
data= pd.read_csv(path) ## data 是dataframe 的数据类型
data.head()
print('\n')
cols = data.shape[1]
X_data = data.iloc[:,:cols-1]#X是所有行，去掉最后一列， 未标准化
y_data = data.iloc[:,cols-1:]#X是所有行，最后一列
X=X_data.values
Y=y_data.values


#归一化
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
scaler = MinMaxScaler()
X_g = scaler.fit_transform(X)
reg=1
alpha =0.001
iters = 100000
from sklearn.model_selection import KFold
# 分离特征和标签
# 设置K折交叉验证的K值
k = 5
# 创建KFold对象
kf = KFold(n_splits=k, shuffle=True, random_state=42)
auc_scores = []
# 遍历K折交叉验证的每一次划分
for train_index, test_index in kf.split(X_g):
    # 分割训练集和测试集
    X_train, X_test = X_g[train_index], X_g[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
feature_dim=2
W=np.zeros((feature_dim,1))
def computeCost(X_train,Y_train,W):
    P=sigmoid(np.dot(X_train,W))
    loss=np.sum(-Y_train*np.log(P)-(1-Y_train)*np.log(1-P))/X_train.shape[0]
    l2=(reg/2)*np.sum(W**2)

    return loss,P
def gradienDecent(W,X_train,alpha,Y_train):
    error=sigmoid(np.dot(X_train,W))-Y_train
    grad=np.dot(X_train.T,error)/X_train.shape[0]
    W-=alpha*grad
    return W

def logisticRegression(X_train,Y_train,alpha,iters,X_test,Y_test):

    feature_dim=X_train.shape[1]
    loss_his = []
    test_loss=[]
    W=np.zeros((feature_dim,1))
    for i in range(iters):
        loss,P=computeCost(X_train,Y_train,W)
        loss_his.append(loss)
        W=gradienDecent(W,X_train,alpha,Y_train)
        test_l,P=computeCost(X_test,Y_test,W)
        test_loss.append(test_l)
    plt.figure(figsize=(4, 2))
    plt.plot(range(1, iters + 1), loss_his, label='训练曲线')
    plt.plot(range(1, iters + 1), test_loss, label='测试曲线')
    plt.title('误差和训练Epoch数')
    plt.legend(loc=2)
    plt.xlabel('迭代次数')
    plt.ylabel('代价')
    plt.legend()
    plt.show()
    return W
W=logisticRegression(X_train,Y_train,alpha,iters,X_test,Y_test)

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
def predict(W, X_test):
    probability = sigmoid(X_test @ W)  # 使用 '@' 运算符进行矩阵乘法
    y_hat = (probability >= 0.5).astype(int)  # 确保 y_hat 是整数类型的一维数组
    return probability, y_hat
from sklearn.metrics import accuracy_score, precision_score
probability, y_hat = predict(W, X_test)
# 确保 Y_test 是一维数组
# 如果 Y_test 是二维的，并且是多分类问题的 one-hot 编码，则需要进一步处理
# 在二分类问题中，Y_test 通常应该是一维的
# 计算准确度
acc = accuracy_score(Y_test, y_hat)  # 不需要转置 y_hat
# 计算精确度
precision = precision_score(Y_test, y_hat)  # 同样不需要转置 y_hat
recall=recall_score(Y_test, y_hat)
f1 = f1_score(Y_test, y_hat)
fpr, tpr, thresholds = roc_curve(Y_test, probability)
roc_auc = auc(fpr, tpr)
print(f"Accuracy: {acc}")
print(f"Precision: {precision}")
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC: {roc_auc:.4f}')
# print(f'Y_test: {Y_test}')
# print(f'probability: {probability }')
# print(f'y_hat: {y_hat }')
# print(fpr)
# print(tpr)
#画出ROC
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
