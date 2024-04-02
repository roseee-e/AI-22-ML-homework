import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

# ��ʵ��ǩ
Y_true = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])

# Ԥ�����
Y_score = np.array([0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35])

# ����PR����
precision, recall, _ = precision_recall_curve(Y_true, Y_score)

# ����ROC����
fpr, tpr, _ = roc_curve(Y_true, Y_score)

# ����AUCֵ
roc_auc = auc(fpr, tpr)
precision_recall_auc = auc(recall, precision)

# ����PR����
plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve (AUC = {0:0.2f})'.format(precision_recall_auc))

# ����ROC����
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

plt.show()