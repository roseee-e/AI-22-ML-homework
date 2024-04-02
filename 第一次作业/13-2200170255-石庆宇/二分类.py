
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np
y_true = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])
y_scores = np.array([0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35])

# 计算PR曲线的值
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# 计算ROC曲线的值
fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)

# 计算PR曲线的AUC值
auc_pr = auc(recall, precision)

# 计算ROC曲线的AUC值
auc_roc = auc(fpr, tpr)

# 绘制PR曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(recall, precision)
plt.plot(recall, precision, 'ro', label=f' (AUC = {auc_pr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.2])
plt.xlim([-0.1, 1.1])
plt.title('PR')
plt.legend(loc="lower left")

# 绘制ROC曲线
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr,color='k')
plt.plot(fpr, tpr, 'ro', lw=2, label=f'(AUC = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.ylim([0.0, 1.05])
plt.xlim([-0.1, 1.1])
plt.title('ROC')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
