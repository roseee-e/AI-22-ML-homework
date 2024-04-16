import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))
def computeCost(X,Y,W):
    P=sigmoid(np.dot(X,W))
    loss=np.sum(-Y*np.log(P)-(1-Y)*np.log(1-P))/X.shape[0]
    return loss,P
def gradientDecent(X,Y,W,alpha):
    error=sigmoid(np.dot(X,W))-Y
    grad=np.dot(X.T,error)/X.shape[0]
    W-=alpha*grad
    return W
def logisticRegression(X,Y,alpha,iters):
    loss_his=[]
    W_his=[]
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss,P=computeCost(X,Y,W)
        loss_his.append(loss)
        W_his.append(W.copy())
        W=gradientDecent(X,Y,W,alpha)
    return loss_his,W_his,W
def testRegression(X,Y,W_his):
    test_loss=[]
    for W in W_his:
        loss,_=computeCost(X,Y,W)
        test_loss.append(loss)
    return test_loss

path = "C:/Users/王子川/Documents/Tencent Files/2369277526/FileRecv/ex2data1.txt"
data = pd.read_csv(path, names=['Exam 1', 'Exam 2', 'Admitted'])
# print(data.head())

positive=data[data['Admitted']==1]
negative=data[data['Admitted']==0]

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

cols = data.shape[1]
X_data = data.iloc[:,:cols-1]
y_data = data.iloc[:,cols-1:]
X_data=(X_data-X_data['Exam 1'].min())/(X_data['Exam 1'].max()-X_data['Exam 1'].min())
X_data=(X_data-X_data['Exam 2'].min())/(X_data['Exam 2'].max()-X_data['Exam 2'].min())

X_data.insert(0, 'Ones', 1)
X=X_data.values
y=y_data.values

train=[]
test=[]
W_train_his=[]
precision_his=[]
recall_his=[]
f1score_his=[]
fpr_his=[]
tpr_his=[]
roc_auc_his=[]

alpha=0.1
iters=12000

# 定义K折交叉验证对象
kf = KFold(n_splits=5,shuffle=True)
# 使用K折交叉验证划分数据集
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    loss_his,W_his,W=logisticRegression(X_train,y_train,alpha,iters)

    test_loss=testRegression(X_test,y_test,W_his)
    test.append(test_loss)
    train.append(loss_his)
    W_train_his.append(W)

    test_predictions = sigmoid(np.dot(X_test, W))
    binary_predictions = (test_predictions >= 0.5).astype(int)
    precision = precision_score(y_test, binary_predictions)
    recall = recall_score(y_test, binary_predictions)
    f1 = f1_score(y_test, binary_predictions)
    fpr, tpr, _ = roc_curve(y_test, test_predictions)

    precision_his.append(precision)
    recall_his.append(recall)
    f1score_his.append(f1)
    fpr_his.append(fpr)
    tpr_his.append(tpr)

avg_train_loss = np.mean(train,axis=0)
avg_test_loss = np.mean(test,axis=0)
W_train_his_avg = np.mean(W_train_his, axis=0)

precision_avg = np.mean(precision_his)
recall_avg = np.mean(recall_his)
f1_avg = np.mean(f1score_his)

fpr_grid = np.linspace(0.0, 1.0, 100)
mean_tpr = np.zeros_like(fpr_grid)
for i in range(5):
    mean_tpr += np.interp(fpr_grid, fpr_his[i], tpr_his[i])
fpr_grid = np.insert(fpr_grid, 0, 0.0)
mean_tpr = np.insert(mean_tpr, 0, 0.0)
mean_tpr = mean_tpr/5

plt.figure(figsize=(5, 3))
plt.plot(np.arange(iters), avg_train_loss, c='r', label='Training Loss')
plt.plot(np.arange(iters), avg_test_loss, c='b', label='Test Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curves')
plt.legend()
plt.show()

plt.figure(figsize=(5, 4))
plt.plot(fpr_grid, mean_tpr, color='red', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()

roc_auc = auc(fpr_grid, mean_tpr)
print('auc=',roc_auc, 'precision_avg=',precision_avg, 'recall_avg=',recall_avg, 'f1_avg=',f1_avg)