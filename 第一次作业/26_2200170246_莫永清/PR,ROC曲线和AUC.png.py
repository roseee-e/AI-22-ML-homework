import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# 给定数据
Y_true = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])
Y_score = np.array([0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35])

# 计算PR曲线
precision, recall, _ = precision_recall_curve(Y_true, Y_score)

# 计算ROC曲线
fpr, tpr, _ = roc_curve(Y_true, Y_score)

# 计算AUC值
auc_score = auc(fpr, tpr)

# 绘制PR曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# 绘制ROC曲线
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

# 显示AUC值
plt.suptitle(f'AUC = {auc_score:.2f}')
plt.tight_layout()
plt.show()