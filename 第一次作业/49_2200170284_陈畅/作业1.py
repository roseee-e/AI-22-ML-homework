import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

Y_true = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])
Y_score = np.array([0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35])
precision, recall, _ = precision_recall_curve(Y_true, Y_score)
fpr, tpr, _ = roc_curve(Y_true, Y_score)
# 计算AUC值
roc_auc = auc(fpr, tpr)
precision_recall_auc = auc(recall, precision)
plt.figure(figsize=(12, 6))
# 绘制PR曲线
plt.subplot(1, 2, 1)
plt.plot(recall, precision, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve (AUC = {0:0.2f})'.format(precision_recall_auc))
# 绘制ROC曲线
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {0:0.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')

plt.show()