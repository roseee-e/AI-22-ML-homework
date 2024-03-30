import numpy as up
import matplotlib.pyplot as plt
T=[[0,0,1],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0]]
P=[[0.1,0.2,0.7],[0.1,0.6,0.3],[0.5,0.2,0.3],[0.1,0.1,0.8],[0.4,0.2,0.4],[0.6,0.3,0.1],[0.4,0.2,0.4],[0.4,0.1,0.5],[0.1,0.1,0.8],[0.1,0.8,0.1]]
P1=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
x=[]
y=[]
x1=[]
y1=[]
tp=[]
fp=[]
TP=[]
FP=[]
fn=[]
FN=[]
TN=[]
tn=[]
count1=0
for i in range(len(T[0])):
    for p in range(len(P1)):
        count = 0
        count1=0
        count2=0
        for p1 in range(len(P)):
            if(P1[p]<=P[p1][i]):
                if(T[p1][i]==0):
                    count1=count1+1
                else:
                    count2=count2+1
        count3=0
        for p2 in range(len(T)):
            if(T[p2][i]==0):
                count3=count3+1
        count4=len(T)-count3
        fn.append(count4-count2)
        tn.append(count3-count1)
        tp.append(count2)
        fp.append(count1)
        x1.append(count1/count3)
        y1.append(count2/count4)
    FN.append(fn)
    TN.append(tn)
    fn=[]
    tn=[]
    TP.append(tp)
    FP.append(fp)
    tp=[]
    fp=[]
    x.append(x1)
    y.append(y1)
    x1=[]
    y1=[]
tpr=[]
fpr=[]
for t in range(10):
    tpr.append((TP[0][t] + TP[1][t] + TP[2][t])/(TP[0][t] + TP[1][t] + TP[2][t]+FN[0][t]+FN[1][t]+FN[2][t]))
    fpr.append((FP[0][t]+FP[1][t]+FP[2][t])/(TN[0][t] + TN[1][t] + TN[2][t]+FP[0][t]+FP[1][t]+FP[2][t]))
x.append(fpr)
y.append(tpr)
for t in [3]:
    plt.plot(x[t],y[t])
plt.show()


