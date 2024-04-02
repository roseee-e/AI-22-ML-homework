import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

# 真实标签和预测概率
y_true = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])
y_score = np.array([0.9, 0.4, 0.2, 0.6, 0.5, 0.4, 0.7, 0.4, 0.65, 0.35])

# 计算ROC曲线的FPR和TPR
fpr, tpr, _ = roc_curve(y_true, y_score)
# 计算ROC曲线的AUC值
roc_auc = roc_auc_score(y_true, y_score)

# 计算PR曲线的Precision和Recall
precision, recall, _ = precision_recall_curve(y_true, y_score)
# 计算PR曲线的AUC值
pr_auc = auc(recall, precision)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 绘制PR曲线
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# 输出ROC和PR曲线的AUC值
print('ROC AUC:', roc_auc)
print('PR AUC:', pr_auc)
