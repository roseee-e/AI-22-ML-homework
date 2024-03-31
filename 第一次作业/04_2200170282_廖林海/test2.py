import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

y_true = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
y_pred = np.asarray([[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4], [0.6, 0.3, 0.1], [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
n_classes = len(y_true[0])

plt.figure(figsize=(10, 8))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, marker='.', label=f'Class {i+1} (AUC = {roc_auc:.2f})')

'''micro'''
fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_pred.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, marker='.', label=f'Micro-average (AUC = {roc_auc_micro:.2f})')

'''macro'''
fpr_grid = np.linspace(0.0, 1.0, 100)
mean_tpr = np.zeros_like(fpr_grid)
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
    mean_tpr += np.interp(fpr_grid, fpr, tpr)
mean_tpr /= n_classes
fpr_macro = fpr_grid
tpr_macro = mean_tpr
roc_auc_macro = auc(fpr_macro, tpr_macro)
plt.plot(fpr_macro, tpr_macro, marker='.', label=f'Macro-average (AUC = {roc_auc_macro:.2f})')

'''Weighted'''
y_true_list = list([tuple(t) for t in y_true])
classNum = dict((a,y_true_list.count(a)) for a in y_true_list)
n1 = classNum[(1, 0, 0)]
n2 = classNum[(0, 1, 0)]
n3 = classNum[(0, 0, 1)]
ratio = [n1/(n1+n2+n3), n2/(n1+n2+n3), n3/(n1+n2+n3)]
avg_tpr = np.zeros_like(fpr_grid)
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
    avg_tpr += ratio[i] * np.interp(fpr_grid, fpr, tpr)
fpr_weighted = fpr_grid
tpr_weighted = avg_tpr
roc_auc_weighted = auc(fpr_weighted, tpr_weighted)
plt.plot(fpr_weighted, tpr_weighted, marker='.', label=f'Weighted-average (AUC = {roc_auc_weighted:.2f})')

plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')

plt.tight_layout()
plt.show()
