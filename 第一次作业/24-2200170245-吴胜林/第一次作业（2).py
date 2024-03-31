import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# 示例数据
Y_true = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])
Y_score = np.array([0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35])

# 计算 PR 曲线
precision, recall, _ = precision_recall_curve(Y_true, Y_score)

# 计算 ROC 曲线
fpr, tpr, _ = roc_curve(Y_true, Y_score)

# 计算 AUC 值
auc_pr = auc(recall, precision)
auc_roc = auc(fpr, tpr)

# 绘制 PR 曲线
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(recall, precision, color='b', label=f'PR curve (AUC = {auc_pr:0.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')

# 绘制 ROC 曲线
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='r', label=f'ROC curve (AUC = {auc_roc:0.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

print(f'AUC PR: {auc_pr:.2f}')
print(f'AUC ROC: {auc_roc:.2f}')
