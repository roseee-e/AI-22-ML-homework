import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve

pp=[[1,0.9],[0,0.4],
    [0,0.2],[1,0.6],
    [0,0.5],[0,0.4],
    [1,0.7],[1,0.4],
    [0,0.65],[0,0.35]
]

y_true=[]
y_score=[]
for p in pp:
    y_c=p[0]
    if y_c==1:
        y=1
    else:
        y=0

    y_hat=p[1]
    y_true.append(y)
    y_score.append(y_hat)


auc=roc_auc_score(y_true,y_score,sample_weight=None)
print(auc)
fpr,tpr,th=roc_curve(y_true,y_score)


plt.plot(fpr,tpr)
plt.title('ROC curve',fontsize=14)
plt.ylabel('TPR',fontsize=14)
plt.xlabel('FPR',fontsize=14)
plt.show()