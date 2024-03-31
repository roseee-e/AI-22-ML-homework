import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

y_true = np.asarray(
    [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0],
     [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])

y_pred = np.asarray(
    [[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4],
     [0.6, 0.3, 0.1], [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])

# 计算类别数量 几列
classes = len(y_true[1, :])
# 初始化字典来存储不同类别的假正率和真正率
fpr = dict()
tpr = dict()
# 初始化字典来记录不同情况下的 ROC AUC
roc_auc = dict()
# 计算每个类别的 ROC 曲线和 ROC AUC
for i in range(classes):
    # 计算当前类别的 ROC 曲线
    fpr[i], tpr[i], th = roc_curve(y_true[:, i], y_pred[:, i])
    # 计算当前类别的 ROC_AUC
    roc_auc[i] = auc(fpr[i], tpr[i])

# ROC曲线
fig, axs = plt.subplots(nrows=2, ncols=classes, figsize=(8, 5))

for i in range(classes):
    axs[0, i].plot(fpr[i], tpr[i])
    axs[0, i].plot(fpr[i], tpr[i], 'ro')
    axs[0, i].set_title('ROC Curve for Class {}'.format(i + 1), fontsize=10)
    axs[0, i].set_xlabel('FPR', fontsize=10)
    axs[0, i].set_ylabel('TPR', fontsize=10)

# 平均ROC曲线

# 微观micro_roc
fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_pred.ravel())
axs[1, 0].plot(fpr_micro, tpr_micro)
axs[1, 0].set_title('Micro_Roc Curve', fontsize=10)
axs[1, 0].set_xlabel('FPR', fontsize=10)
axs[1, 0].set_ylabel('TPR', fontsize=10)
roc_auc_micro = auc(fpr_micro, tpr_micro)

# 宏观macro_roc
fpr_grid = np.linspace(0.0, 1.0,100)
mean_tpr = np.zeros_like(fpr_grid)
for i in range(classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])

mean_tpr /= classes
roc_auc_macro = auc(fpr_grid, mean_tpr)
# fpr_macro = np.add.reduce([fpr[i] for i in range(classes)]) / classes
# tpr_macro = np.add.reduce([tpr[i] for i in range(classes)]) / classes
axs[1, 1].plot(fpr_grid, mean_tpr)
axs[1, 1].set_title('Macro_Roc Curve', fontsize=10)
axs[1, 1].set_xlabel('FPR', fontsize=10)
axs[1, 1].set_ylabel('TPR', fontsize=10)


# 加权weighted_average_roc
weights = np.array([sum(y_true[:, i]) for i in range(classes)], dtype=np.float64)
# 计算权重
weights /= sum(weights)
# 计算加权平均
# fpr_weighted = np.average([fpr[i] for i in range(classes)], axis=0, weights=weights)
# tpr_weighted = np.average([tpr[i] for i in range(classes)], axis=0, weights=weights)
avg_tpr = np.zeros_like(fpr_grid)
for i in range(classes):
    avg_tpr += weights[i]*np.interp(fpr_grid, fpr[i], tpr[i])

roc_auc_weighted = auc(fpr_grid, avg_tpr)
axs[1, 2].plot(fpr_grid, avg_tpr)
axs[1, 2].set_title('Weighted_Average_Roc Curve', fontsize=10)
axs[1, 2].set_xlabel('FPR', fontsize=10)
axs[1, 2].set_ylabel('TPR', fontsize=10)

plt.tight_layout()
plt.show()
