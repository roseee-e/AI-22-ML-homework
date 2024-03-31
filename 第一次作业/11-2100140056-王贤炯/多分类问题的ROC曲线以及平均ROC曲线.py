import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, auc

y_true = np.asarray(
    [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
y_pred = np.asarray(
    [[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4], [0.6, 0.3, 0.1],
     [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])

# print(y_true.shape,y_pred.shape)
# (10, 3) (10, 3)

n_class = len(y_true[1, :])
# n_class表示取y_true的第1行中所有列的个数,即类别数
fpr = dict()
tpr = dict()
roc_auc = dict()
plt.figure(figsize=(10, 10))
for i in range(n_class):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    # 将y_true和y_pred中，遍历列，将对应列算出的fpr和tpr存入字典中
    roc_auc[i] = auc(fpr[i], tpr[i])
    print("AUC(第", i, "列)=", roc_auc[i])
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0}'.format(i))
    plt.plot(fpr[i], tpr[i], 'ro')

plt.xlabel('FPR', fontsize=16)
plt.ylabel('TPR', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc="lower right")
# 循环作出每个类别的ROC曲线

fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
#
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
print("AUC(micro)=", roc_auc["micro"])

plt.figure(figsize=(10, 10))
plt.plot(fpr["micro"], tpr["micro"])
plt.plot(fpr["micro"], tpr["micro"])
plt.xlabel('FPR', fontsize=16)
plt.ylabel('TPR', fontsize=16)
plt.title('ROC curve(micro)', fontsize=16)
# 作平均ROC曲线（micro）

fpr_grid = np.linspace(0, 1, 100)

mean_tpr = np.zeros_like(fpr_grid)
for i in range(n_class):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
    # 利用插值，取循环将每个类别在各自的ROC图中的点的tpr值，再累加起来

mean_tpr /= n_class
# 累加的tpr值再取平均，就是macro的tpr值

fpr["macro"] = fpr_grid
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
print("AUC(macro)=", roc_auc["macro"])

plt.figure(figsize=(10, 10))
plt.plot(fpr["macro"], tpr["macro"])
plt.xlabel('FPR', fontsize=16)
plt.ylabel('TPR', fontsize=16)
plt.title('ROC curve(macro)', fontsize=16)
# 作平均ROC曲线（macro）

y_true_list = list([tuple(t) for t in y_true])
# 将y_true数组转成元组，在转成列表，存在y_true_list中
classNum = dict((a, y_true_list.count(a)) for a in y_true_list)
# 将y_true_list转成字典
n1 = classNum[(1, 0, 0)]
n2 = classNum[(0, 1, 0)]
n3 = classNum[(0, 0, 1)]

nall = n1 + n2 + n3  # 总类别数
ratio = [n1 / nall, n2 / nall, n3 / nall]
avg_tpr = np.zeros_like(fpr_grid)
for i in range(n_class):
    avg_tpr += ratio[i] * np.interp(fpr_grid, fpr[i], tpr[i])

fpr["weighted"] = fpr_grid
tpr["weighted"] = avg_tpr
roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])
print("AUC(weighted)=", roc_auc["weighted"])

plt.figure(figsize=(10, 10))
plt.plot(fpr["weighted"], tpr["weighted"])
plt.xlabel('FPR', fontsize=16)
plt.ylabel('TPR', fontsize=16)
plt.title('ROC curve(weighted)', fontsize=16)
#  作平均ROC曲线（weighted））

plt.show()
