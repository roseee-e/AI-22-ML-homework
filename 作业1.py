import numpy as up
import matplotlib.pyplot as plt
Y_true=[1,0,0,1,0,0,1,1,0,0]
Y_score=[0.9,0.4,0.2,0.6,0.5,0.4,0.7,0.4,0.65,0.35]
a=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]
true1=sum(Y_true)
true0=len(Y_true)-true1
tp=0
fp=0
Tp=[]
Fp=[]
Fn=[]
Tn=[]
P=[]
R=[]
Tpr=[]
Fpr=[]
Auc=0
for j in range(10):
    tp=0
    fp=0
    for i in range(10):
        if(a[j]<=Y_score[i]):
            if(Y_true[i]==1):
                tp=tp+1
            else:
                fp=fp+1
    Tp.append(tp)
    Fp.append(fp)
    Fn.append(true1-tp)
    Tn.append(true0-fp)
for i in range(10):
    P.append(Tp[i]/(Tp[i]+Fp[i]))
    R.append(Tp[i]/(Tp[i]+Fn[i]))
    Fpr.append(Fp[i]/(Tn[i]+Fp[i]))
Tpr=R
for i in range(9):
    Auc=(Fpr[i+1]-Fpr[i])*(Tpr[i]+Tpr[i+1])/2+Auc
print(Auc)
plt.plot(P,R)
plt.plot(Fpr,Tpr)
plt.show()