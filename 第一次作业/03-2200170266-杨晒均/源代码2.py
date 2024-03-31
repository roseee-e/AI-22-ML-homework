
import matplotlib.pyplot as plt
import numpy as np
# 首先对得到的数据进行降序排序
pp = [['T', 0.9], ['N', 0.4], ['N', 0.2], ['T', 0.6], ['N', 0.5], ['N', 0.4], ['T', 0.7], ['T', 0.4], ['N', 0.65], ['N', 0.35]]
pp.sort(key=lambda x: x[1], reverse=True)
aa = [0.9, 0.4, 0.2, 0.6, 0.5, 0.4, 0.7, 0.4, 0.65, 0.35]
aa.sort(reverse=True)
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
    for p in pp:
        if p[0] == 'T' and p[1] >= a:
            tp += 1
        elif p[0] == 'T' and p[1] < a:
            fn += 1
        elif p[0] == 'N' and p[1] >= a:
            fp += 1
        elif p[0] == 'N' and p[1] < a:
            tn += 1
    x = float(tp) / (tp + fn)
    y = float(tp) / (tp + fp)
    if tn + fp == 0:
        fpr = 0
    else:
        fpr = float(fp) / (tn + fp)
    recall.append(x)
    precision.append(y)
    TPR.append(x)
    FPR.append(fpr)
# 绘制 ROC 曲线图
plt.figure(1)
plt.figure(figsize=(5, 5))
plt.title('ROC curve', fontsize=14)
plt.plot(FPR, TPR)
plt.plot(FPR, TPR, 'ro')
plt.ylabel('TPR', fontsize=16)
plt.xlabel('FPR', fontsize=16)
# 绘制 Precision-Recall 曲线图
plt.figure(2)
plt.figure(figsize=(5, 5))
plt.title('Precision-Recall curve', fontsize=16)
plt.plot(recall, precision)
plt.plot(recall, precision, 'ro')
plt.ylabel('Precision', fontsize=16)
plt.xlabel('Recall', fontsize=16)
# 显示图像
plt.show()
i = 0
auc = 0
while i < 9:
    auc += (FPR[i + 1] - FPR[i]) * (TPR[i] + TPR[i + 1])
    i += 1
auc = float(auc / 2)
print('auc=%.2f' % auc)

