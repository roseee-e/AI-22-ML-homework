import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 每个样本的真实标签和预测得分
Y_true = np.array([[0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 1, 0]])

Y_score = np.array([[0.1, 0.2, 0.7],
                     [0.1, 0.6, 0.3],
                     [0.5, 0.2, 0.3],
                     [0.1, 0.1, 0.8],
                     [0.4, 0.2, 0.4],
                     [0.6, 0.3, 0.1],
                     [0.4, 0.2, 0.4],
                     [0.4, 0.1, 0.5],
                     [0.1, 0.1, 0.8],
                     [0.1, 0.8, 0.1]])

# 计算每个类别的ROC曲线
plt.figure()
for i in range(3):
    fpr, tpr, _ = roc_curve(Y_true[:, i], Y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = {:.2f}) for Class {}'.format(roc_auc, i+1))

# 计算微平均（micro-average）ROC曲线
fpr_micro, tpr_micro, _ = roc_curve(Y_true.ravel(), Y_score.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, label='Micro-average ROC curve (area = {:.2f})'.format(roc_auc_micro), linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class and Micro-average')
plt.legend(loc="lower right")
plt.show()
