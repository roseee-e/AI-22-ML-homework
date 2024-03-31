import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
import numpy as np

# 数据
pp = np.array([[1, 0.90], [1, 0.70], [0, 0.65],[1, 0.60], [0, 0.50],[0, 0.40],  [0, 0.40],
                [1, 0.40], [0, 0.35],[0, 0.20] ])

y_true = pp[:, 0]
y_scores = pp[:, 1]

# PR曲线
precision, recall, _ = precision_recall_curve(y_true, y_scores)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title('Precision-Recall Curve', fontsize=16)
plt.plot(recall, precision)
plt.plot(recall, precision, 'ro')
plt.ylabel('Precision', fontsize=16)
plt.xlabel('Recall', fontsize=16)

# ROC曲线
auc = roc_auc_score(y_true, y_scores, sample_weight=None)
fpr, tpr, th = roc_curve(y_true, y_scores)

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr)
plt.plot(fpr, tpr,'ro')
plt.title("ROC curve", fontsize=14)
plt.ylabel('TPR', fontsize=14)
plt.xlabel('FPR', fontsize=14)

# 显示AUC值
print("AUC值", auc)

plt.show()
