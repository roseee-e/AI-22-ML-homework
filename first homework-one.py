import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# 模拟数据
Y_true = np.array([1,0,0,1,0,0,1,1,0,0])
Y_score = np.array([0.9,0.4,0.2,0.6,0.5,0.4,0.7,0.4,0.65,0.35])

precision, recall, _ = precision_recall_curve(Y_true, Y_score)
fpr, tpr, _ = roc_curve(Y_true, Y_score)

# 计算AUC值
pr_auc = auc(recall, precision)
roc_auc = auc(fpr, tpr)

# 画PR曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (AUC = %0.2f)' % pr_auc)
plt.grid(True)

# 画ROC曲线
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve (AUC = %0.2f)' % roc_auc)
plt.grid(True)

plt.tight_layout()
plt.show()
