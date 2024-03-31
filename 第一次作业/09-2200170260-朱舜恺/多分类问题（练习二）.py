import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

y_true = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0],
                     [0, 1, 0], [0, 0, 1], [0, 1, 0]])
y_pred = np.asarray([[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4],
                     [0.6, 0.3, 0.1], [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])

print(y_true.shape, y_pred.shape)      # number of samples * number of classes

n_classes = len(y_true[1, :])        # Computer ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])

fpr_grid = np.linspace(0.0, 1.0, 100)

y_true_list = list([tuple(t) for t in y_true])
classNum = dict((a, y_true_list.count(a)) for a in y_true_list)
n1 = classNum[(1, 0, 0)]
n2 = classNum[(0, 1, 0)]
n3 = classNum[(0, 0, 1)]
ratio = [n1 / (n1 + n2 + n3), n2 / (n1 + n2 + n3), n3 / (n1 + n2 + n3)]
avg_tpr = np.zeros_like(fpr_grid)  # t have the same dimension with fprs.

fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())

fpr_grid = np.linspace(0.0, 1.0, 100)
mean_tpr = np.zeros_like(fpr_grid)
for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = fpr_grid
tpr["macro"] = mean_tpr

for i in range(n_classes):
    avg_tpr += ratio[i] * np.interp(fpr_grid, fpr[i], tpr[i])    # get the corresponding tprs with linear interpolations

fpr["weighted"] = fpr_grid
tpr["weighted"] = avg_tpr

plt.figure(figsize=(6, 6))

plt.plot(fpr["macro"], tpr["macro"], linestyle=':', linewidth=2)
plt.plot(fpr["micro"], tpr["micro"], linestyle=':', linewidth=2)
plt.plot(fpr["weighted"], tpr["weighted"], linestyle=':', linewidth=2)

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i])

plt.title('ROC curve', fontsize=14)
plt.ylabel('TPR', fontsize=14)
plt.xlabel('FPR', fontsize=14)
plt.legend(["micro-average ROC curve", "macro-average ROC curve", "weighted average ROC curve",
            "ROC curve of class 0", "ROC curve of class 1", "ROC curve of class 2"])
plt.show()
