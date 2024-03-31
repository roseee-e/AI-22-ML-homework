import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

y_true = np.asarray(
    [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0],[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
y_pred = np.asarray(
    [[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4],
     [0.6, 0.3, 0.1], [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])

n_classes = len(y_true[1, :])
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    # 计算当前类别的 ROC 曲线
    fpr[i], tpr[i], th = roc_curve(y_true[:, i], y_pred[:, i])
    # 计算当前类别的 ROC_AUC
    roc_auc[i] = auc(fpr[i], tpr[i])
# ROC曲线

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i])
    plt.title('ROC Curve{}'.format(i + 1))
    plt.xlabel('FPR')
    plt.ylabel('TPR')

# 平均ROC曲线

# 微观micro_roc
fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_pred.ravel())
plt.plot(fpr_micro, tpr_micro)
plt.title('Micro_Roc_Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
roc_auc_micro = auc(fpr_micro, tpr_micro)

# 宏观macro_roc
fpr_grid = np.linspace(0.0, 1.0,100)
mean_tpr = np.zeros_like(fpr_grid)
for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])

mean_tpr /= n_classes
roc_auc_macro = auc(fpr_grid, mean_tpr)
plt.plot(fpr_grid, mean_tpr)
plt.title('Macro_Roc_Curve')
plt.xlabel('FPR')
plt.set_ylabel('TPR')

# 加权
weights = np.array([sum(y_true[:, i]) for i in range(n_classes)], dtype=np.float64)
weights /= sum(weights)
# 加权平均
avg_tpr = np.zeros_like(fpr_grid)
for i in range(n_classes):
    avg_tpr += weights[i]*np.interp(fpr_grid, fpr[i], tpr[i])
roc_auc_weighted = auc(fpr_grid, avg_tpr)
plt.plot(fpr_grid, avg_tpr)
plt.title('Weighted_Average_Roc Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
