import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt

Y_true = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])
Y_score = np.array([0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35])


precision, recall, thresholds = precision_recall_curve(Y_true, Y_score)
pr_auc = auc(recall, precision)


fpr, tpr, roc_thresholds = roc_curve(Y_true, Y_score)
roc_auc = auc(fpr, tpr)


print("Precision-Recall AUC:", pr_auc)
print("ROC AUC:", roc_auc)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(recall, precision, color='r',  lw=2,label='PR curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('PR curve')
plt.legend(loc="lower left")


plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.legend(loc="lower right")

plt.show()
