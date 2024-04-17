
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
path = r'C:\Users\lenovo\Downloads\ex2data1.csv'
data = pd.read_csv(path)
X_data= data.iloc[:, :2].values
y_data= data.iloc[:, 2].values
x=np.append(X_data,np.ones((X_data.shape[0],1)),axis=1)
w=np.zeros((x.shape[1],1)).T


def normalize(x):
    x1=(x-np.min(x))/(np.max(x)-np.min(x))
    return x1


def sigmoid(z):
    return 1/(1+np.exp(-z))



def computecost(x,y,w):
    z=sigmoid(w@x.T)
    loss=np.sum(-y*np.log(z)-(1-y)*np.log(1-z))/x.shape[0]
    return loss,z



def gradientdecent(w,x,y):
    error=sigmoid(w@x.T)-y
    grad=(error@x)/x.shape[0]
    w-=alpha*grad
    return w



def logisticregression(x,y,alpha,iters,w):
    for i in range(iters):
        loss,p=computecost(x,y,w)
        loss_his.append(loss)
        w=gradientdecent(w,x,y)
    return loss_his,w
loss_his=[]
alpha=0.0001
iters=1000
w=np.zeros((x.shape[1],1)).T
all_precision = 0
all_recall = 0
all_f1 = 0
all_auc=0
fpr_list = []
tpr_list = []
loss_sum = []
testloss_sum = []
W_models = []



#K折
kf = KFold(n_splits=5, shuffle=True, random_state=30)
for train_index, test_index in kf.split(X, Y):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    train_loss_history, W_train, W_his = logisticregression(x_train, y_train, alpha, iters,w)
    W_his.append(W_train)
    test_loss_history = []
    for W in W_his:
        loss, _ = computeCost(x_test, y_test, W)
        test_loss_history.append(loss)
    train_loss.append(train_loss_history)
    test_loss.append(test_loss_history)
    # 评估模型性能
    confusion_matrix, precision, recall, f1_score, fpr, tpr, roc_auc = evaluate_performance(W_train, x_test,
                                                                                                    y_test)


#计算平均
loss_average = np.mean(loss_sum, axis=0)
testloss_average =np.mean(testloss_sum, axis=0)
precision_average=all_precision/5
recall_average=all_recall/5
f1_average=all_f1/5
max_len = max(len(fpr) for fpr in fpr_list)
extended_fpr_list = [np.concatenate([fpr, np.full(max_len - len(fpr), np.nan)]) for fpr in fpr_list]
extended_tpr_list = [np.concatenate([tpr, np.full(max_len - len(tpr), np.nan)]) for tpr in tpr_list]
fpr_average = np.nanmean(extended_fpr_list, axis=0)
tpr_average = np.nanmean(extended_tpr_list, axis=0)
auc_average=all_auc/5



# 画出图
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
for i in range(len(fpr_his)):
    plt.plot(fpr_his[i], tpr_his[i], label='ROC(面积 = %0.2f)' % roc_auc_his[i])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('每折的ROC曲线')
plt.legend()
plt.show()
plt.plot(avg_test_loss,label='测试')
plt.plot(avg_train_loss,label='训练')
plt.title('训练损失函数')
plt.xlabel('次数')
plt.ylabel('损失')
plt.legend()
plt.show()


# 输出性能指标
print(f"精确度: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")
print(f"AUC值: {auc:.4f}")


