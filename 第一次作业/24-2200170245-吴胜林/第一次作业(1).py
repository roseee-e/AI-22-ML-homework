from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

# 示例数据
true_labels = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
predict_scores = np.array([[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4], [0.6, 0.3, 0.1], [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])

# 计算每个类别的 ROC 曲线和 ROC AUC 分数
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predict_scores[:, i])
    roc_auc[i] = roc_auc_score(true_labels[:, i], predict_scores[:, i])

# 计算 micro 平均
micro_fpr, micro_tpr, _ = roc_curve(true_labels.ravel(), predict_scores.ravel())
micro_roc_auc = roc_auc_score(true_labels.ravel(), predict_scores.ravel(), average='micro')

# 计算 macro 平均
macro_fpr = np.linspace(0, 1, 100)
macro_tpr = np.zeros_like(macro_fpr)
for i in range(3):
    macro_tpr += np.interp(macro_fpr, fpr[i], tpr[i])
macro_tpr /= 3
macro_roc_auc = roc_auc_score(true_labels, predict_scores, average='macro')

# 绘制每个类别的 ROC 曲线和平均 ROC 曲线
plt.figure()
plt.plot(micro_fpr, micro_tpr, label='micro-average ROC curve (area = {0:0.2f})'.format(micro_roc_auc), color='deeppink', linestyle=':')
plt.plot(macro_fpr, macro_tpr, label='macro-average ROC curve (area = {0:0.2f})'.format(macro_roc_auc), color='navy', linestyle=':')

for i in range(3):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver Operating Characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()