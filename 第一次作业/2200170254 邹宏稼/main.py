
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

Y_true = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])
Y_score = np.array([0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35])


precision, recall, thresholds = precision_recall_curve(Y_true, Y_score)
pr_auc = auc(recall, precision)


fpr, tpr, roc_thresholds = roc_curve(Y_true, Y_score)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(recall, precision, color='b', label='PR curve (area = %0.2f)' % pr_auc)
plt.plot(recall, precision,'ro')
plt.xlabel('召回率')
plt.ylabel('精度')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.05])
plt.title('PR曲线')
plt.legend(loc="lower left")


plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fpr, tpr, 'ro')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('ROC曲线')
plt.legend(loc="lower right")

plt.show()
