import matplotlib.pyplot as plt
import numpy as np
pp=[['1',0.90],['0',0.40],['0',0.20],['1',0.60],['0',0.50],
    ['0',0.40],['1',0.70],['1',0.40],['0',0.65],['0',0.35]]
aa=[0.90,0.70,0.65,0.60,0.50,0.40,0.40,0.40,0.35,0.20]

recall=[]
precision=[]
TPR=[]
FPR=[]

for a in aa:
    tp=0
    fn=0
    fp=0
    tn=0
    x=0
    y=0
    for p in pp:
        if(p[0]=='1')and(p[1]>=a):
            tp=tp+1
        elif(p[0]=='1')and(p[1]<a):
            fn=fn+1
        elif(p[0]=='0')and(p[1]>=a):
            fp=fp+1
        elif(p[0]=='0')and(p[1]<a):
            tn=tn+1

    x=float(tp)/(tp+fn)
    y=float(tp)/(tp+fp)
    fpr=float(fp)/(tn+fp)

    recall.append(x)
    precision.append(y)
    TPR.append(x)
    FPR.append(fpr)
plt.figure(figsize=(5,5))
plt.subplot(2,1,1)
plt.title('precision-recall curve',fontsize=16)
plt.plot(recall,precision)
plt.plot(recall,precision,'ro')
plt.ylabel('Precision',fontsize=16)
plt.xlabel('Recall',fontsize=16)

plt.subplot(2,1,2)

plt.title('ROC curve',fontsize=14)
plt.plot(FPR,TPR)
plt.plot(FPR,TPR,'ro')
plt.ylabel('TPR',fontsize=14)
plt.xlabel('FPR',fontsize=14)
plt.show()

i=0
auc=0
while(i<9):
    auc=auc+(FPR[i+1]-FPR[i])*(TPR[i+1]+TPR[i])
    i=i+1
auc=float(auc/2)
print('auc=%.2f' % auc)