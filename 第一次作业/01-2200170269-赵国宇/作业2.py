import numpy as np
from sklearn.metrics import roc_curve, auc,roc_auc_score
import matplotlib.pyplot as plt

# 真实标签
y_true = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])

# 预测概率
y_pred = np.asarray([[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4], [0.6, 0.3, 0.1], [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])

# 类别数量
n_classes = len(y_true[0])

# 计算每个类别的 ROC 曲线和 AUC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 将标签二值化
y_test_binarized = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])

# 计算 micro-average ROC 曲线和 ROC 面积
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 计算 macro-average ROC 曲线和 ROC 面积
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# 计算 weighted-average ROC 曲线和 ROC 面积
roc_auc["weighted"] = roc_auc_score(y_test_binarized, y_pred, average="weighted")

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# 子图1: 每个类别的 ROC 曲线
colors = ['blue', 'green', 'red']
for i, color in zip(range(n_classes), colors):
    ax1.plot(fpr[i], tpr[i], color=color, lw=2,
             label='Class {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))
ax1.set_title('ROC curve per class')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(loc="lower right")

# 子图2: 三种平均 ROC 曲线
ax2.plot(fpr["micro"], tpr["micro"],
         label='Micro-average (AUC = {0:0.2f})'.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
ax2.plot(fpr["macro"], tpr["macro"],
         label='Macro-average (AUC = {0:0.2f})'.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
ax2.set_title('Average ROC curves')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.legend(loc="lower right")

# 显示图表
plt.show()

# 输出 weighted-average AUC
print('Weighted-average AUC: {0:0.2f}'.format(roc_auc["weighted"]))