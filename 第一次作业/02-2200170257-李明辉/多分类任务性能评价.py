import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


Y_true = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                           [0, 0, 1], [0, 1, 0]])
Y_score = np.asarray([[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4],
                               [0.6, 0.3, 0.1], [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])


num_classes = Y_true.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()


for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_true[:, i], Y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             label='ROC curve{0}'
                   ''.format(i, roc_auc[i]))




micro_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
micro_tpr = np.zeros_like(micro_fpr)
for i in range(num_classes):
    micro_tpr += np.interp(micro_fpr, fpr[i], tpr[i])
micro_tpr /= num_classes
micro_roc_auc = auc(micro_fpr, micro_tpr)
plt.plot(micro_fpr, micro_tpr,
         label='micro-average ROC curve'
               ''.format(micro_roc_auc),
         color='deeppink', linestyle='--', linewidth=2)




plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()