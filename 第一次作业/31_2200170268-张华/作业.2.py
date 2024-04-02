import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 真实标签和预测概率，转换为NumPy数组
true = np.array(
    [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
predict = np.array(
    [[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4], [0.6, 0.3, 0.1],
     [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])

# 初始化图形窗口和子图
fig, ax = plt.subplots(figsize=(10, 8))

# 计算每个类别的ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(true[:, i], predict[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算平均FPR和TPR
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(3):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= 3

# 绘制所有类别的ROC曲线
for i in range(3):
    ax.plot(fpr[i], tpr[i], label=f'Class {i + 1} (AUC = {roc_auc[i]:.2f})')

# 绘制平均ROC曲线
ax.plot(all_fpr, mean_tpr, label=f'Average (AUC = {roc_auc[0]:.2f})', linestyle='--', color='navy')

# 设置图形的标题和坐标轴标签
ax.set_title('Receiver Operating Characteristic (ROC) Curves')
ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR)')
ax.legend(loc="lower right")

# 显示图形
plt.show()