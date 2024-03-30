import matplotlib.pyplot as plt
import  numpy as np
from sklearn.metrics import precision_recall_curve,roc_curve,auc

sample = [['T',0.9],['N',0.4],['N',0.2],['T',0.6],['N',0.5],
          ['N',0.4],['T',0.7],['T',0.4],['N',0.65],['N',0.35]]
Y_true = [1,0,0,1,0,0,1,1,0,0]
Y_score = [0.9,0.4,0.2,0.6,0.5,0.4,0.7,0.4,0.65,0.35]

pr,re,th = precision_recall_curve(Y_true,Y_score)

plt.figure(figsize=(5, 5))
plt.title('Precision-Recall Curve',fontsize = 16)
plt.plot(re,pr)
plt.plot(re,pr,'ro')
plt.xlabel('Recall',fontsize = 16)
plt.ylabel('Precision',fontsize = 16)

plt.show()

fpr,tpr,rth = roc_curve(Y_true, Y_score)
auc = auc(fpr,tpr)

plt.plot(fpr,tpr)
plt.title('Roc Curve',fontsize = 14)
plt.xlabel('Fpr',fontsize = 14)
plt.ylabel('Tpr',fontsize = 14)

plt.show()

print(auc)