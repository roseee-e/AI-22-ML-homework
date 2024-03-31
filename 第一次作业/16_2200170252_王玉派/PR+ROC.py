import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

pp1 = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])
aa1 = np.array([0.9, 0.4, 0.2, 0.6, 0.5, 0.4, 0.7, 0.4, 0.65, 0.35])

# 根据预测概率的降序排列索引
index = np.argsort(aa1)[::-1]
# 根据排序后的索引重新排列
pp = pp1[index]
aa = aa1[index]

# 求PR
recall = []
precision = []
TPR = []
FPR = []

for a in aa:
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    x = 0
    y = 0
    for p in range(len(pp)):
        if (pp[p] == 1) and (aa[p] >= a):
            tp = tp + 1
        elif (pp[p] == 1) and (aa[p] < a):
            fn = fn + 1
        elif (pp[p] == 0) and (aa[p] >= a):
            fp = fp + 1
        elif (pp[p] == 0) and (aa[p] < a):
            tn = tn + 1
    x = float(tp) / (tp + fn)
    y = float(tp) / (tp + fp)
    fpr = float(fp) / (tn + fp)

    recall.append(x)
    precision.append(y)
    TPR.append(x)
    FPR.append(fpr)

# 展示PR曲线
plt.figure(figsize=(5,5))
plt.title('precision-recall curve',fontsize=16)
plt.plot(recall,precision)
plt.plot(recall,precision,'ro')
plt.ylabel('Precision',fontsize=16)
plt.xlabel('Recall',fontsize=16)
plt.show()

# 求ROC
y_true = []
y_score = []
for p in range(len(pp)):
    y_c = pp[p]
    if y_c == 1:
        y = 1
    else:
        y = 0

    y_hat = aa[p]
    y_true.append(y)
    y_score.append(y_hat)

# 计算auc
auc = roc_auc_score(y_true, y_score, sample_weight=None)
# print(auc)
fpr, tpr, th = roc_curve(y_true, y_score)

# 展示ROC
'''plt.plot(fpr,tpr)
plt.title('ROC curve',fontsize=14)
plt.ylabel('TPR',fontsize=14)
plt.xlabel('FPR',fontsize=14)
plt.show()'''
