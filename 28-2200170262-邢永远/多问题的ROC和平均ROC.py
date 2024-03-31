import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

y_true = np.asarray([[0,0,1],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0]])
y_pred = np.asarray([[0.1,0.2,0.7], [0.1,0.6,0.3], [0.5,0.2,0.3], [0.1,0.1,0.8], [0.4,0.2,0.4],[0.6,0.3,0.1],
                     [0.4,0.2,0.4], [0.4,0.1,0.5],[0.1,0.1,0.8],[0.1,0.8,0.1]])
n_classes=len(y_true[1,:])
fpr=dict()
tpr=dict()
roc_auc=dict()

for i in range(n_classes):
    fpr[i],tpr[i],_=roc_curve(y_true[:,i],y_pred[:,i])
    #roc_auc[i]=auc(fpr[i],tpr[i])
for i in range(n_classes):
    plt.title('ROC curve', fontsize=14)
    plt.plot( fpr[i],tpr[i])
    plt.plot(fpr[i],tpr[i],'ro')
    plt.ylabel(f'TPR', fontsize=14)
    plt.xlabel(f'FPR', fontsize=14)


fpr_grid=np.linspace(0.0,1.0,100)
mean_tpr=np.zeros_like(fpr_grid)
for i in range(n_classes):
    mean_tpr+=np.interp(fpr_grid,fpr[i],tpr[i])
mean_tpr/=n_classes

fpr["macro"]=fpr_grid
tpr["macro"]=mean_tpr
plt.plot( fpr["macro"],tpr["macro"])
plt.show()