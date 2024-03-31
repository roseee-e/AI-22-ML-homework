import sys
sys.path.append("D:\python\lib\site-packages")
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_true = np.asarray([[0,0,1],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0]])
y_pred = np.asarray([[0.1,0.2,0.7],[0.1,0.6,0.3],[0.5,0.2,0.3],[0.1,0.1,0.8],[0.4,0.2,0.4],[0.6,0.3,0.1],
                    [0.4,0.2,0.4],[0.4,0.1,0.5],[0.1,0.1,0.8],[0.1,0.8,0.1]])
n_classes=len(y_true[1,:])
# 计算每个类别的ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制每个类别的ROC曲线
plt.figure()
colors = ['r', 'g', 'b']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (auc = {1:0.2f})'
             ''.format(i, roc_auc[i]))
    plt.plot(fpr[i], tpr[i],'ro')
# 绘制平均ROC曲线
# Micro平均

fpr_grid=np.linspace(0.0,1.0,5)

mean_tpr = np.zeros_like(fpr_grid)
for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = fpr_grid
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (auc = {0:0.2f})'
         ''.format(roc_auc["macro"]),
         color='k', linestyle='-', linewidth=2)
plt.plot(fpr["macro"], tpr["macro"],'ro')
# 设置图形参数
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('test2')
plt.legend(loc="lower right")
plt.show()
