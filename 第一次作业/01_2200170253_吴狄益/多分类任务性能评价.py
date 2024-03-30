import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc

y_true = np.asarray([[0,0,1],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0]])
y_pred = np.asarray([[0.1,0.2,0.7],[0.1,0.6,0.3],[0.5,0.2,0.3],[0.1,0.1,0.8],[0.4,0.2,0.4],[0.6,0.3,0.1],
[0.4,0.2,0.4],[0.4,0.1,0.5],[0.1,0.1,0.8],[0.1,0.8,0.1]])

n_classes = len(y_true[1,:])
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i],tpr[i],_ = roc_curve(y_true[:,i],y_pred[:,i])
    roc_auc[i] = auc(fpr[i],tpr[i])

plt.figure(figsize=(10,8))
for i in range(3):
    plt.plot(fpr[i],tpr[i])
    plt.title('Roc Curve',fontsize=14)
    plt.xlabel('Fpr',fontsize=14)
    plt.ylabel('Tpr',fontsize=14)

fpr["micro"],tpr["micro"],_ = roc_curve(y_true.ravel(),y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"],tpr["micro"])

plt.plot(fpr["micro"],tpr["micro"])
plt.plot(fpr["micro"],tpr["micro"],'ro')
plt.title('Roc Curve',fontsize=14)
plt.xlabel('Fpr',fontsize=14)
plt.ylabel('Tpr',fontsize=14)
plt.legend(['No.1_Roc Curve','No.2_Roc Curve','No.3_Roc Curve','Average Roc Curve'])
plt.show()