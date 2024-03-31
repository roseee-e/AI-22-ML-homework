import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


y_true = np.asarray([[0,0,1],[0,1,0],[1,0,0],[0,0,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0]])
y_pred = np.asarray([[0.1,0.2,0.7],[0.1,0.6,0.3],[0.5,0.2,0.3],[0.1,0.1,0.8],[0.4,0.2,0.4],[0.6,0.3,0.1],
                     [0.4,0.2,0.4],[0.4,0.1,0.5],[0.1,0.1,0.8],[0.1,0.8,0.1]])

n_classes = len(y_true[0])

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i],_ = roc_curve(y_true[:, i], y_pred[:, i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i],label= f'Class {i+1} ')
    plt.plot(fpr["micro"], tpr["micro"],'k-')
plt.plot(fpr["micro"], tpr["micro"],'k-',label='avg curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) curves')
plt.legend(loc='lower right')
plt.grid()
plt.legend()
plt.show()