import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 定义数据
Y_true = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0],
                    [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
Y_score = np.array([[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4],
                     [0.6, 0.3, 0.1], [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])

# 计算每个类别的ROC曲线
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(Y_true.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(Y_true[:, i], Y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算微平均、宏平均和加权平均的ROC曲线
# 微平均
fpr_micro, tpr_micro, _ = roc_curve(Y_true.ravel(), Y_score.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# 宏平均
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(Y_true.shape[1])]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(Y_true.shape[1]):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= Y_true.shape[1]
fpr_macro = all_fpr
tpr_macro = mean_tpr
roc_auc_macro = auc(fpr_macro, tpr_macro)

# 加权平均
sample_count = np.sum(Y_true, axis=0)
weighted_sum_tpr = np.zeros_like(all_fpr)
for i in range(Y_true.shape[1]):
    weighted_sum_tpr += np.interp(all_fpr, fpr[i], tpr[i]) * sample_count[i]
weighted_avg_tpr = weighted_sum_tpr / np.sum(sample_count)
fpr_weighted_avg = all_fpr
tpr_weighted_avg = weighted_avg_tpr
roc_auc_weighted_avg = auc(fpr_weighted_avg, tpr_weighted_avg)

# 绘制每个类别的ROC曲线
plt.figure()
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
for i in range(Y_true.shape[1]):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {} (area = {:0.2f})'.format(i, roc_auc[i]))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curves')
plt.legend(loc="lower right")
plt.show()

# 绘制平均ROC曲线
plt.figure()
plt.plot(fpr_micro, tpr_micro, color='deeppink', lw=2, linestyle=':', label='Micro-average ROC curve (area = {:0.2f})'.format(roc_auc_micro))
plt.plot(fpr_macro, tpr_macro, color='navy', lw=2, linestyle=':', label='Macro-average ROC curve (area = {:0.2f})'.format(roc_auc_macro))
plt.plot(fpr_weighted_avg, tpr_weighted_avg, color='darkorange', lw=2, linestyle=':', label='Weighted-average ROC curve (area = {:0.2f})'.format(roc_auc_weighted_avg))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Average ROC curves')
plt.legend(loc="lower right")
plt.show()
