import numpy as np
from sklearn.metrics 
import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

Y_true = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])

Y_score = np.array([0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35])

precision, recall, _ = precision_recall_curve(Y_true, Y_score)

pr_auc = auc(recall, precision)

fpr, tpr, _ = roc_curve(Y_true, Y_score)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(recall, precision, label=f'PR Curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')

plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.tight_layout()
plt.show()