import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# 初始化数据
pp = [['T', 0.9], ['N', 0.4], ['N', 0.2], ['T', 0.6], ['N', 0.5], ['N', 0.4], ['T', 0.7], ['T', 0.4], ['N', 0.65], ['N', 0.35]]
aa = [0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.3, 0.1]

# 初始化列表
recall = []
precision = []
TPR = []
FPR = []

# 遍历不同的阈值
for a in aa:
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    # 遍历每个实例
    for p in pp:
        if (p[0] == 'T') and (p[1] >= a):
            tp += 1
        elif (p[0] == 'T') and (p[1] < a):
            fn += 1
        elif (p[0] == 'N') and (p[1] >= a):
            fp += 1
        elif (p[0] == 'N') and (p[1] < a):
            tn += 1
    # 计算指标
    x = float(tp) / (tp + fn) if (tp + fn) > 0 else 0
    y = float(tp) / (tp + fp) if (tp + fp) > 0 else 0
    fpr = float(fp) / (tn + fp) if (tn + fp) > 0 else 0
    recall.append(x)
    precision.append(y)
    TPR.append(x)
    FPR.append(fpr)

# 准备 ROC 曲线和 AUC 的数据
y_true = []
y_score = []

for p in pp:
    y_c = p[0]
    y = 1 if y_c == 'T' else 0
    y_hat = p[1]
    y_true.append(y)
    y_score.append(y_hat)

# 计算 AUC 和绘制 ROC 曲线
auc_score = roc_auc_score(y_true, y_score)
fpr, tpr, thresholds = roc_curve(y_true, y_score)

# 创建一个图形和一组子图
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# 在第一个子图上绘制 PR 曲线
axs[0].plot(recall, precision, label='PR Curve')
axs[0].scatter(recall, precision)  # 画出点
axs[0].set_title('Precision-Recall Curve', fontsize=16)
axs[0].set_xlabel('Recall', fontsize=16)
axs[0].set_ylabel('Precision', fontsize=16)
axs[0].legend(loc='lower left')

# 在第二个子图上绘制 ROC 曲线
axs[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
axs[1].set_title('ROC Curve', fontsize=16)
axs[1].set_xlabel('False Positive Rate', fontsize=16)
axs[1].set_ylabel('True Positive Rate', fontsize=16)
axs[1].legend(loc='lower right')

# 显示图形
plt.tight_layout()
plt.show()
