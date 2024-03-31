import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

y_true = np.asarray(
    [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0],
     [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
y_pred = np.asarray(
    [[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4], [0.6, 0.3, 0.1],
     [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
fpr = {}
tpr = {}
roc_auc = {}
for i in range(len(y_true[1, :])):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i], pos_label=1)
    roc_auc[i] = auc(fpr[i], tpr[i])
plt.figure(figsize=(10, 10))
for i in range(3):
    plt.subplot(2, 2, i+1)
    plt.plot(fpr[i], tpr[i], c='c', lw=3)
    plt.title('roc curve', fontsize=14)
    plt.plot(fpr[i], tpr[i], 'ro')
    plt.ylabel('fpr', fontsize=14)
    plt.xlabel('tpr', fontsize=14)


fpr_grid = np.linspace(0.0, 1.0, 100)
y_true_list = list([tuple(t) for t in y_true])
classNum = dict((a, y_true_list.count(a)) for a in y_true_list)
n1 = classNum[(1, 0, 0)]
n2 = classNum[(0, 1, 0)]
n3 = classNum[(0, 0, 1)]
ratio = [n1/(n1+n2+n3), n2/(n1+n2+n3), n3/(n1+n2+n3)]
avg_tpr = np.zeros_like(fpr_grid)
for i in range(3):
    avg_tpr += ratio[i]*np. interp(fpr_grid, fpr[i], tpr[i])
    fpr["weighted"] = fpr_grid
    tpr["weighted"] = avg_tpr
    roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])

plt.subplot(2, 2, 4)
plt.plot(fpr["weighted"], tpr["weighted"], c='c', lw=2)
plt.title('aver_roc curve', fontsize=14)
plt.ylabel('fpr', fontsize=14)
plt.xlabel('tpr', fontsize=14)
plt.show()
