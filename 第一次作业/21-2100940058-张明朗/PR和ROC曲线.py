import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# 实际标签
Y_true = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])
# 预测得分
Y_score = np.array([0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35])

# 计算PR曲线
precision, recall, _ = precision_recall_curve(Y_true, Y_score)
# 计算AUC值
pr_auc = auc(recall, precision)

# 计算ROC曲线
fpr, tpr, _ = roc_curve(Y_true, Y_score)
# 计算AUC值
roc_auc = auc(fpr, tpr)

#显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']

# 绘制PR曲线和ROC曲线
plt.figure(figsize=(12, 6))

# 绘制PR曲线
plt.subplot(1, 2, 1)
plt.plot(recall, precision, color='blue', lw=2, label='PR 曲线 (area = %0.2f)' % pr_auc)
plt.plot(recall, precision,"ro")
plt.xlabel('R')
plt.ylabel('P')
plt.title('PR曲线')
plt.legend(loc='lower right')
print('PR AUC: %0.2f' % pr_auc)

# 绘制ROC曲线
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='red', lw=2, label='ROC 曲线 (area = %0.2f)' % roc_auc)
plt.plot(fpr, tpr,"ro")
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC曲线')
plt.legend(loc='lower right')
print('ROC AUC: %0.2f)' % roc_auc)

plt.tight_layout()
plt.show()
