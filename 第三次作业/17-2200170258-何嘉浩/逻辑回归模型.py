import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams 
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler  
from sklearn.metrics import precision_score,recall_score,f1_score,roc_curve,auc

config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,  
    'axes.unicode_minus': False 
}
rcParams.update(config)  

path = 'E:/QQ文件/2369152757/FileRecv/ex2data1.txt'

#读取数据
import pandas as pd
data = pd.read_csv(path,header=None,names=['Exam1','Exam2','Admit']) 
cols = data.shape[1]
rows = data.shape[0]
positive_data=data[data['Admit']==1]
negative_data=data[data['Admit']==0]

#数据归一化
scaler = MinMaxScaler()  
X_normalized = scaler.fit_transform(data.iloc[:,0:2])
X_normalized = np.concatenate((np.ones((X_normalized.shape[0], 1)),X_normalized), axis=1)
positive_data = scaler.fit_transform(positive_data.iloc[:,0:2])
positive_data = np.concatenate((np.ones((positive_data.shape[0], 1)),positive_data), axis=1)
negative_data = scaler.fit_transform(negative_data.iloc[:,0:2])
negative_data = np.concatenate((np.ones((negative_data.shape[0], 1)),negative_data), axis=1)
Y1=data.iloc[:,2:3]
Y=Y1.values

#画散点图

fig,ax = plt.subplots(figsize=(8,6))
ax.scatter(positive_data[:,1], positive_data[:,2], c='b', marker='o', label='positive')
ax.scatter(negative_data[:,1], negative_data[:,2], c='r', marker='x', label='negative')
plt.legend()
plt.xlabel('train 1 Score')
plt.ylabel('train 2 Score')
plt.title('Data distribution')
plt.show()

#编写函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

def computeCost(X, Y, W):
    P=sigmoid(np.dot(X,W)) 
    loss=np.sum(-Y*np.log(P)-(1-Y)*np.log(1-P))/X.shape[0]
    return loss,P

def gradientDescent(X, Y, W, alpha): 
   error=sigmoid(np.dot(X,W))-Y
   dW=np.dot(X.T,error)/X.shape[0]
   W=W-alpha*dW
   return W

def logisticRegression(X,Y, alpha,iters):
    loss_his = []
    W_his=[]
    feature_dim = X.shape[1]
    W=np.zeros((feature_dim,1)) 
    for i in range (iters):
        loss,P = computeCost(X,Y,W)
        loss_his.append(loss)
        W=gradientDescent(X, Y, W, alpha)
        W_his.append(W)
    return loss_his, W ,W_his

def test(X,Y,W,iters):
    loss_his_test = []
    for i in range (iters):
        loss_test,P = computeCost(X,Y,W[i])
        loss_his_test.append(loss_test)
    return loss_his_test

#全局变量
alpha =0.1
iters = 10000
loss_his_test=[]
loss_his_train=[]
W_aver=[]

#K折交叉验证
kfold = KFold(n_splits=10,shuffle=True,random_state=0)
for train_index,test_index in kfold.split(X_normalized):
    #训练模型
    W_his=[]
    train_X,train_Y = X_normalized[train_index],Y[train_index]
    test_X,test_Y = X_normalized[test_index],Y[test_index]
    loss_his, W,W_his= logisticRegression(train_X, train_Y, alpha, iters)
    #测试模型
    test_loss = test(test_X,test_Y,W_his,iters)
    loss_his_train.append(loss_his)
    loss_his_test.append(test_loss)
    W_aver.append(W)

#求数据均值
train_loss_aver=np.mean(loss_his_train,axis=0)
test_loss_aver=np.mean(loss_his_test,axis=0)
W_aver1=np.mean(W_aver,axis=0)

#换损失函数
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(len(train_loss_aver)), train_loss_aver, 'r',label='训练集损失函数')
ax.plot(np.arange(len(test_loss_aver)), test_loss_aver, 'g',label='测试集损失函数')
ax.legend(loc=1)
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('逻辑回归模型损失函数')
plt.show()


def predict(X, W):
    '''
    输入：
        X：测试数据集
        W：模型训练好的参数
    输出：
        y_pre：预测值
    '''
    Y_pre = sigmoid(X@W)
    Y_hat=Y_pre>=0.5
    return Y_pre,Y_hat

#求预测值
Y_pre,Y_hat=predict(X_normalized,W_aver1)
Y_hat=Y_hat+0
Y_hat=Y_hat.T
Y_true=Y.T
from itertools import chain
Y_true = list(chain.from_iterable(Y_true))
Y_hat = list(chain.from_iterable(Y_hat))

#求presicion和recall、F1值
precision=precision_score(Y_true,Y_hat)
print("prescision值为{}".format(precision))
recall=recall_score(Y_true,Y_hat)
print("recall值为{}".format(recall))
F1=f1_score(Y_true,Y_hat)
print("f1_score值为{}".format(F1))


#画ROC曲线和求AUC
fpr,tpr,thre=roc_curve(Y_true,Y_hat,)
AUC=auc(fpr,tpr)
plt.figure(figsize=(5, 5))  
plt.title('ROC Curve',fontsize=16)  
plt.plot(fpr, tpr)  
plt.plot(fpr, tpr,'ro')  
plt.xlabel('tpr',fontsize=16)  
plt.ylabel('fpr',fontsize=16)  
plt.show()
print("AUC为{}".format(AUC))




