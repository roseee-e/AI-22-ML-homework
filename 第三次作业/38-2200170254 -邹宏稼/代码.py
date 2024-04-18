import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

config = {
        "mathtext.fontset": 'stix',
        "font.family": 'serif',
        "font.serif": ['SimHei'],
        "font.size": 10,
        'axes.unicode_minus': False
    }
rcParams.update(config)

path = r"E:\qq\ex2data1.txt"
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])#手动指定header
pdData.head()
X_data= pdData.iloc[:, :2]
Y_data= pdData.iloc[:, 2]
##print(Y_data.head)
##print(pdData.head())

positive=pdData[pdData['Admitted']==1]
negative=pdData[pdData['Admitted']==0]
##print(positive)
##print(negative)
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(positive['Exam 1'],positive['Exam 2'],s=30,c='b',marker='o',label='Admitted')
ax.scatter(negative['Exam 1'],negative['Exam 2'],s=30,c='r',marker='x',label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))
##归一
def normalize_data(X_data):
    X_min = np.min(X_data)
    X_max = np.max(X_data)
    X_narm=(X_data-X_min)/(X_max-X_min)
    return X_narm

X_narm=normalize_data(X_data)
X_narm= np.insert(X_narm, 0, 1, axis=1)
X=X_narm
Y=Y_data.values.reshape(-1, 1)
##print(Y)
feature_dim=X.shape[1]
W=np.zeros((feature_dim,1))

def computeCost(X, Y, W):
    P = sigmoid(np.dot(X, W))
    loss = np.sum(-Y * np.log(P) - (1 - Y) * np.log(1 - P)) / len(Y)
    return loss, P

def gradientDecent(W,X,Y,a):
    error=sigmoid(np.dot(X,W))-Y
    grad=np.dot(X.T,error)/len(Y)
    W=W-a*grad
    return  W

def logisticRegression(X,Y,alpha,iters):
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    loss_his = []
    W_his = []
    for i in range(iters):
        loss,P=computeCost(X,Y,W)
        loss_his.append(loss)
        W_his.append(W.copy())
        W=gradientDecent(W,X,Y,alpha)
    return loss_his,W_his,W

alpha=0.04
iters=10000

def testmodel(X, Y, W_his, iters):
    testloss_his = []
    for i in range(min(iters, len(W_his))):
        loss, P = computeCost(X, Y, W_his[i])
        testloss_his.append(loss)
    return testloss_his,P

precision_sum= 0
recall_sum = 0
f1_sum = 0
auc_sum= 0
fpr_sum = []
tpr_sum = []
loss_sum = []
testloss_sum = []
W_models = []
kf = KFold(n_splits=5, shuffle=True, random_state=None)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    loss_his,W_his, W = logisticRegression(X_train, Y_train, alpha, iters)
    testloss_his, P = testmodel(X_test, Y_test, W_his, iters)
    W_models.append(W)
    loss_sum.append(loss_his)
    testloss_sum.append(testloss_his)
    precision = precision_score(Y_test, np.round(P),zero_division=1)
    recall = recall_score(Y_test, np.round(P),zero_division=1)
    f1 = f1_score(Y_test, np.round(P),zero_division=1)
    precision_sum += precision
    recall_sum += recall
    f1_sum += f1
    fpr, tpr, _ = roc_curve(Y_test, P)
    roc_auc = auc(fpr, tpr)
    fpr_sum.append(fpr)
    tpr_sum.append(tpr)
    auc_sum+=roc_auc
##print(loss_his)

loss_aver = np.mean(loss_sum,axis=0)
test_loss_aver = np.mean(testloss_sum,axis=0)
##print(loss_sum)
precision_aver = np.mean(precision_sum)
recall_aver= np.mean(recall_sum)
f1_aver = np.mean(f1_sum)
fpr_aver = np.linspace(0.0, 1.0, 100)
tpr_aver = np.zeros_like(fpr_aver)
for i in range(5):
    tpr_aver += np.interp(fpr_aver, fpr_sum[i], tpr_sum[i])
fpr_aver = np.insert(fpr_aver, 0, 0.0)
tpr_aver = np.insert(tpr_aver, 0, 0.0)
tpr_aver = tpr_aver / 5
auc_aver = auc_sum / 5


plt.figure(figsize=(6, 4))
plt.plot(np.arange(len(loss_aver)), loss_aver, 'b',label='训练集损失函数')
plt.plot(np.arange(len(test_loss_aver)), test_loss_aver, 'g',label='测试集损失函数')
#print(loss_aver)
##print(test_loss_aver)
plt.xlabel('迭代次数')
plt.ylabel('代价', rotation=0)
plt.title('训练和测试损失函数')
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(fpr_aver, tpr_aver, 'r', label='ROC Curve')
plt.plot([0, 1], [0, 1], 'g', linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('平均ROC曲线')
plt.legend()
plt.show()

print("precision_aver:",precision_aver)
print("recall_aver:",recall_aver)
print("f1_aver:",f1_aver)
print("auc_aver:",auc_aver)