import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

Y_true = [1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
Y_score = [0.9, 0.4, 0.2, 0.6, 0.5, 0.4, 0.7, 0.4, 0.65, 0.35]

precision, recall, _ = precision_recall_curve(Y_true, Y_score)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve: AUC={0:0.2f}'.format(pr_auc))

fpr, tpr, _ = roc_curve(Y_true, Y_score)
roc_auc = auc(fpr, tpr)

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve: AUC={0:0.2f}'.format(roc_auc))

plt.tight_layout()
plt.show()

print("PR AUC:", pr_auc)
print("ROC AUC:", roc_auc)
