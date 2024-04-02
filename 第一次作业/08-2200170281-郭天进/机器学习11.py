import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

Y_true = [1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
Y_score = [0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35]

precision, recall, _ = precision_recall_curve(Y_true, Y_score)

fpr, tpr, _ = roc_curve(Y_true, Y_score)

auc_pr = auc(recall, precision)
auc_roc = auc(fpr, tpr)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('P-R(AUC = {:.2f})'.format(auc_pr))
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC(AUC = {:.2f})'.format(auc_roc))
plt.grid(True)

#plt.tight_layout()
plt.show()
