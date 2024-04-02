import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import numpy as np

samples = np.array([
    [[0, 0, 1], [0.1, 0.2, 0.7]],
    [[0, 1, 0], [0.1, 0.6, 0.3]],
    [[1, 0, 0], [0.5, 0.2, 0.3]],
    [[0, 0, 1], [0.1, 0.1, 0.8]],
    [[1, 0, 0], [0.4, 0.2, 0.4]],
    [[0, 1, 0], [0.6, 0.3, 0.1]],
    [[0, 1, 0], [0.4, 0.2, 0.4]],
    [[0, 1, 0], [0.4, 0.1, 0.5]],
    [[0, 0, 1], [0.1, 0.1, 0.8]],
    [[0, 1, 0], [0.1, 0.8, 0.1]]
])

# 计算每个类别的真正例率和假正例率
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(samples[:, 0, i], samples[:, 1, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 画ROC曲线
plt.figure()

colors = ['red', 'blue', 'green']
labels = ['Class 1', 'Class 2', 'Class 3']

for i in range(3):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=labels[i] + ' ROC curve (area = %0.2f)' % roc_auc[i])

# 绘制均值ROC曲线
# 可以选择使用micro或macro或weighted average等方法计算平均ROC曲线

# 计算micro平均ROC曲线
micro_fpr, micro_tpr, _ = roc_curve(samples[:, 0, :].ravel(), samples[:, 1, :].ravel())
micro_roc_auc = auc(micro_fpr, micro_tpr)

# 绘制micro平均ROC曲线
plt.plot(micro_fpr, micro_tpr, color='deeppink', lw=2,
         label='Micro-average ROC curve (area = %0.2f)' % micro_roc_auc)

# 计算macro平均ROC曲线
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
mean_tpr = np.average([np.interp(all_fpr, fpr[i], tpr[i]) for i in range(3)], axis=0)
macro_roc_auc = auc(all_fpr, mean_tpr)

# 绘制macro平均ROC曲线
plt.plot(all_fpr, mean_tpr, color='navy', lw=2,
         label='Macro-average ROC curve (area = %0.2f)' % macro_roc_auc)

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
