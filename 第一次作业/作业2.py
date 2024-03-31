import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_true = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
y_pred = np.asarray([[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4], [0.6, 0.3, 0.1], [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])

n_classes = len(y_true[0])

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(8, 6))

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (class %d) AUC = %.2f' % (i, roc_auc[i]))

#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multiple Classes')
plt.legend(loc="lower right")

# Compute macro-average ROC curve and ROC area
fpr_grid = np.linspace(0.0, 1.0, 100)  # Given all the possible FPRs, ranging from 0 to 1, defining x-axis
# Interpolate all ROC curves at these points
mean_tpr = np.zeros_like(fpr_grid)  # Have the same dimension with fprs
for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # Get the corresponding TPRs with linear interpolations
# Average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = fpr_grid
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.plot(fpr["macro"], tpr["macro"], lw=2, linestyle='-', label='Macro-average ROC curve (AUC = %.2f)' % roc_auc["macro"])

plt.legend(loc="lower right")  # Add legend for the macro-average ROC curve
plt.show()
