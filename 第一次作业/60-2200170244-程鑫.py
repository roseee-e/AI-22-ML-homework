import matplotlib.pyplot as plt
from sklearn.metrics import auc,precision_recall_curve, roc_curve

Y_true = [1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
Y_score = [0.9, 0.4, 0.2, 0.6, 0.5, 0.4, 0.7, 0.4, 0.65, 0.35]
p, r,_ = precision_recall_curve(Y_true, Y_score)
pr_auc = auc(r, p)
fpr, tpr,_ = roc_curve(Y_true, Y_score)
roc_auc = auc(fpr, tpr)
plt.subplot(1, 2, 1)
plt.plot(r, p)
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr)
plt.show()
