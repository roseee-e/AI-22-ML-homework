import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
def Classfy(a,b):#方便组合数据
    x=[]
    for i in range(0,len(a)):
        x.append([a[i],b[i]])
    return x
y_true=np.asarray([1,0,0,1,0,0,1,1,0,0])#样本真值
y_score=np.asarray([0.9,0.4,0.2,0.6,0.5,0.4,0.7,0.4,0.65,0.35])#分类器对样本的预测值
task=Classfy(y_true,y_score)
Tpr=[]
Fpr=[]
Pre=[]
for i in range(9,0,-1):
    i=i/10
    tn=0
    tp=0
    fn=0
    fp=0
    for k in task:
        if k[1]>=i:
            if k[0]==1:
                tp+=1
            else:
                fp+=1
        else:
            if k[0]==1:
                fn+=1
            else:
                tn+=1
    tpr=float(tp)/(tp+fn)
    fpr=float(fp)/(fp+tn)
    pre=float(tp)/(tp+fp)
    Pre.append(pre)
    Tpr.append(tpr)
    Fpr.append(fpr)
#ROC曲线的点对
point_ROC_Curve=Classfy(Fpr,Tpr)
#PR曲线的点对
point_PR_Curve=Classfy(Tpr,Pre)
#画PR曲线
plt.plot(Tpr,Pre)
plt.xlabel('Precision')
plt.ylabel('Recall')
#画ROC曲线
plt.plot(Fpr,Tpr)
plt.xlabel('Fpr')
plt.ylabel('Tpr')
plt.show()
#计算ROC曲线的AUC
area=0
for i in range(0,len(point_ROC_Curve)-1):
    x=point_ROC_Curve[i+1][0]-point_ROC_Curve[i][0]
    y = point_ROC_Curve[i + 1][1] + point_ROC_Curve[i][1]
    area+=x*y/2
print(area)




