import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

y_true = np.asarray([[0,0,1],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0]])
y_pred = np.asarray([[0.1,0.2,0.7],[0.1,0.6,0.3],[0.5,0.2,0.3],[0.1,0.1,0.8],[0.4,0.2,0.4],[0.6,0.3,0.1],
[0.4,0.2,0.4],[0.4,0.1,0.5],[0.1,0.1,0.8],[0.1,0.8,0.1]])

n_classes = len(y_true[1,:])
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.subplot(2,2,1)
plt.plot(fpr["micro"], tpr["micro"],
         label='Micro Average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
         linestyle=':', linewidth=3)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
# Compute macro-average ROC curve and ROC area
fpr_grid = np.linspace(0.0,1.0,100)
mean_tpr = np.zeros_like(fpr_grid)
for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = fpr_grid
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
plt.subplot(2,2,2)
plt.plot(fpr["macro"], tpr["macro"],
         label='Macro Average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
         linestyle=':', linewidth=3)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
#weighted
fpr_grid = np.linspace(0.0,1.0,100)
# class ratios
y_true_list = list([tuple(t) for t in y_true])
classNum = dict((a, y_true_list.count(a))for a in y_true_list)
n1 = classNum[(1,0,0)]
n2 = classNum[(0,1,0)]
n3 = classNum[(0,0,1)]
ratio = [n1/(n1+n2+n3),n2/(n1+n2+n3),n3/(n1+n2+n3)]
avg_tpr = np.zeros_like(fpr_grid)## have the same dimension with fprs.
for i in range(n_classes):
    avg_tpr += ratio[i]*np.interp(fpr_grid, fpr[i], tpr[i]) # get the corresponding tprs with linear interpolations
fpr["weighted"] = fpr_grid
tpr["weighted"] = avg_tpr
roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])


plt.subplot(2,2,3)
plt.plot(fpr["weighted"], tpr["weighted"],
         label='weighted Average ROC curve (area = {0:0.2f})'.format(roc_auc["weighted"]),
         linestyle=':', linewidth=3)

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.subplot(2,2,4)

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()


