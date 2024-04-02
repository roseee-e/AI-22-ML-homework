import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

y_true = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
y_pred = np.asarray([[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4], [0.6, 0.3, 0.1], [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])

classes = len(y_true[1, :])
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(classes):
    fpr[i], tpr[i], th = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fig, axs = plt.subplots(nrows=2, ncols=classes, figsize=(8, 5))
for i in range(classes):
    axs[0, i].plot(fpr[i], tpr[i])
    axs[0, i].plot(fpr[i], tpr[i], 'ro')
    axs[0, i].set_title('Class {} Curve'.format(i + 1), fontsize=14)
    axs[0, i].set_xlabel('FPR', fontsize=14)
    axs[0, i].set_ylabel('TPR', fontsize=14)

fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_pred.ravel())
axs[1, 0].plot(fpr_micro, tpr_micro)
axs[1, 0].set_title('Micro_Roc Curve', fontsize=14)
axs[1, 0].set_xlabel('FPR', fontsize=14)
axs[1, 0].set_ylabel('TPR', fontsize=14)
roc_auc_micro = auc(fpr_micro, tpr_micro)

fpr_grid = np.linspace(0.0, 1.0, 100)
mean_tpr = np.zeros_like(fpr_grid)
for i in range(classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])

mean_tpr /= classes
roc_auc_macro = auc(fpr_grid, mean_tpr)

axs[1, 1].plot(fpr_grid, mean_tpr)
axs[1, 1].set_title('Macro_Roc Curve', fontsize=14)
axs[1, 1].set_xlabel('FPR', fontsize=14)
axs[1, 1].set_ylabel('TPR', fontsize=14)

weights = np.array([sum(y_true[:, i]) for i in range(classes)], dtype=np.float64)
weights /= sum(weights)

avg_tpr = np.zeros_like(fpr_grid)
for i in range(classes):
    avg_tpr += weights[i] * np.interp(fpr_grid, fpr[i], tpr[i])

roc_auc_weighted = auc(fpr_grid, avg_tpr)
axs[1, 2].plot(fpr_grid, avg_tpr)
axs[1, 2].set_title('Weighted_Average_Roc Curve', fontsize=10)
axs[1, 2].set_xlabel('FPR', fontsize=14)
axs[1, 2].set_ylabel('TPR', fontsize=14)

plt.tight_layout()
plt.show()
