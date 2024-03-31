import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
def Classfy(a, b, i):
    x = []
    for j in range(len(a)):
        x.append([a[j][i], b[j][i]])
    return x
y_true=np.asarray([[0,0,1],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0]])
y_pred=np.asarray([[0.1,0.2,0.7],[0.1,0.6,0.3],[0.5,0.2,0.3],[0.1,0.1,0.8],[0.4,0.2,0.4],[0.6,0.3,0.1],[0.4,0.2,0.4],[0.4,0.1,0.5],[0.1,0.1,0.8],[0.1,0.8,0.1]])
#计算【1 0 0】的tpr与fpr
predict_label1Mark=Classfy(y_true,y_pred,0)
tpr1=[]
fpr1=[]
tp1=[]
tn1=[]
fn1=[]
fp1=[]
for i in range(9,0,-1):
    i=i/10
    tp=0
    fn=0
    tn=0
    fp=0
    x=0
    y=0
    for k in predict_label1Mark:
            if k[1]>i:
                if k[0]==1:
                    tp=tp+1
                else:
                    fp=fp+1
            else:
                if k[0]==1:
                    fn=fn+1
                else:
                    tn=tn+1
    tp1.append(tp)
    tn1.append(tn)
    fn1.append(fn)
    fp1.append(fp)
    x=float(tp)/(tp+fn)
    fpr=float(fp)/(fp+tn)
    tpr1.append(x)
    fpr1.append(fpr)
plt.plot(fpr1,tpr1)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('[1 0 0]roc curve')
#[0 1 0]
predict_label2Mark=Classfy(y_true,y_pred,1)
tpr2=[]
fpr2=[]
tp2=[]
tn2=[]
fn2=[]
fp2=[]
for i in range(9,0,-1):
    i=i/10
    tp=0
    fn=0
    tn=0
    fp=0
    x=0
    for k in predict_label2Mark:
            if k[1]>i:
                if k[0]==1:
                    tp=tp+1
                else:
                    fp=fp+1
            else:
                if k[0]==1:
                    fn=fn+1
                else:
                    tn=tn+1
    tp2.append(tp)
    tn2.append(tn)
    fn2.append(fn)
    fp2.append(fp)
    x=float(tp)/(tp+fn)
    fpr=float(fp)/(fp+tn)
    tpr2.append(x)
    fpr2.append(fpr)
# plt.plot(fpr2,tpr2)
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.title('[0 1 0]roc curve')
#plt.show()
#[0 0 1]
predict_label3Mark=Classfy(y_true,y_pred,2)
tpr3=[]
fpr3=[]
tp3=[]
tn3=[]
fn3=[]
fp3=[]
for i in range(9,0,-1):
    i=i/10
    tp=0
    fn=0
    tn=0
    fp=0
    x=0
    y=0
    for k in predict_label3Mark:
            if k[1]>i:
                if k[0]==1:
                    tp=tp+1
                else:
                    fp=fp+1
            else:
                if k[0]==1:
                    fn=fn+1
                else:
                    tn=tn+1
    tp3.append(tp)
    tn3.append(tn)
    fn3.append(fn)
    fp3.append(fp)
    x=float(tp)/(tp+fn)
    fpr=float(fp)/(fp+tn)
    tpr3.append(x)
    fpr3.append(fpr)
# plt.plot(fpr3, tpr3)
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.title('[0 0 1]roc curve')
#Average Roc curve
allTp=[]
allFp=[]
allFn=[]
allTn=[]
allTpr=[]
allFpr=[]
# micro
for i in range(0,len(tp1)):
    allTn.append(tn1[i]+tn2[i]+tn3[i])
    allFn.append( fn1[i] + fn2[i] + fn3[i])
    allTp.append( tp1[i] + tp2[i] + tp3[i])
    allFp.append( fp1[i] + fp2[i] + fp3[i])
for i in range(0,len(tp1)):
    allTpr.append(float(allTp[i])/(allTp[i]+allFn[i]))
    allFpr.append( float(allFp[i]) / (allFp[i] + allTn[i]))
plt.plot(fpr1,tpr1)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('[1 0 0]roc curve')
plt.plot(fpr2,tpr2)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('[0 1 0]roc curve')
plt.plot(fpr3, tpr3)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('[0 0 1]roc curve')
plt.plot(allFpr,allTpr)
plt.title("aaa")
plt.show()