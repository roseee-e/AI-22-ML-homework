import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 模拟数据
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

# 定义类别标签
classes = ['Class 0', 'Class 1', 'Class 2']

# 初始化变量用于存储每个类别的fpr和tpr
all_fpr = {}
all_tpr = {}
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

# 计算每个类别的ROC曲线并绘制
plt.figure(figsize=(10, 6))
for i in range(3):
    fpr, tpr, _ = roc_curve(Y_true[:, i], Y_score[:, i])
    all_fpr[i] = fpr
    all_tpr[i] = tpr
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC Curve of ' + classes[i] + ' (AUC = %0.2f)' % roc_auc)

# 计算micro ROC曲线
fpr_micro, tpr_micro, _ = roc_curve(Y_true.ravel(), Y_score.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, label='Micro-average ROC Curve (AUC = %0.2f)' % roc_auc_micro, linestyle='--')

# 计算macro ROC曲线
for i in range(3):
    mean_tpr += np.interp(mean_fpr, all_fpr[i], all_tpr[i])
mean_tpr /= 3
roc_auc_macro = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, label='Macro-average ROC Curve (AUC = %0.2f)' % roc_auc_macro, linestyle='--')

# 计算weighted average ROC曲线
w_aucs = []
for i in range(3):
    w_aucs.append(auc(all_fpr[i], all_tpr[i]))
weights = [sum(Y_true[:, i]) for i in range(3)]
weighted_auc = np.average(w_aucs, weights=weights)
plt.plot(mean_fpr, mean_tpr, label='Weighted-average ROC Curve (AUC = %0.2f)' % weighted_auc, linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for each Class and Averaged')
plt.legend()
plt.show()