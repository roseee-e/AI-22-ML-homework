import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
true_labels = np.array([[0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 1, 0]])


predict_scores = np.array([[0.1, 0.2, 0.7],
                           [0.1, 0.6, 0.3],
                           [0.5, 0.2, 0.3],
                           [0.1, 0.1, 0.8],
                           [0.4, 0.2, 0.4],
                           [0.6, 0.3, 0.1],
                           [0.4, 0.2, 0.4],
                           [0.4, 0.1, 0.5],
                           [0.1, 0.1, 0.8],
                           [0.1, 0.8, 0.1]])


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predict_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


all_true_labels = true_labels.ravel()
all_predict_scores = predict_scores.ravel()
fpr["micro"], tpr["micro"], _ = roc_curve(all_true_labels, all_predict_scores)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure()
plt.plot(fpr["micro"], tpr["micro"], label='Micro Average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':')
colors = ['b', 'g', 'r']
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.show()
