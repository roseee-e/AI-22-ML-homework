import matplotlib.pyplot as plt
import numpy as np

pp=[[1,0.9],[0,0.4],
    [0,0.2],[1,0.6],
    [0,0.5],[0,0.4],
    [1,0.7],[1,0.4],
    [0,0.65],[0,0.35]
]
aa=[0.9,0.4,0.2,0.6,0.5,0.4,0.7,0.4,0.65,0.35]

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
        if(p[0]==1) and (p[1]>=a):
            tp=tp+1
        elif(p[0]==1) and (p[1]<a):
            fn=fn+1
        elif(p[0]==0) and (p[1]>=a):
            fp=fp+1
        elif(p[0]==0) and (p[1]<a):
            tn=tn+1
    x=float(tp)/(tp+fn)
    y=float(tp)/(tp+fp)
    fpr=float(fp)/(tn+fp)

    recall.append(x)
    precision.append(y)
    TPR.append(x)
    FPR.append(fpr)

plt.figure(figsize=(5,5))
plt.title('precision-recall curve',fontsize=16)
plt.plot(recall,precision)
plt.plot(recall,precision,'ro')
plt.ylabel('Precision',fontsize=16)
plt.xlabel('Recall',fontsize=16)
plt.show()