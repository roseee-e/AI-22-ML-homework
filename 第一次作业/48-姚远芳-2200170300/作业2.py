# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:52:01 2024

@author: Administrator
"""

import matplotlib.pyplot as plt

data = [['T', 0.9], ['N', 0.4], ['N', 0.2], ['T', 0.6], ['N', 0.5], ['N', 0.4], ['T', 0.7], ['T', 0.4], ['N', 0.65], ['N', 0.35]]
data.sort(key=lambda x: x[1], reverse=True)

thresholds = sorted(set([d[1] for d in data]), reverse=True)
recall = []
precision = []
TPR = []
FPR = []
auc = 0


for threshold in thresholds:
    tp, fn, fp, tn = 0, 0, 0, 0
    
    for d in data:
        if d[0] == 'T' and d[1] >= threshold:
            tp += 1
        elif d[0] == 'T' and d[1] < threshold:
            fn += 1
        elif d[0] == 'N' and d[1] >= threshold:
            fp += 1
        elif d[0] == 'N' and d[1] < threshold:
            tn += 1
    
    r = tp / (tp + fn) if tp + fn != 0 else 0
    p = tp / (tp + fp) if tp + fp != 0 else 0
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp) if tn + fp != 0 else 0
    
    recall.append(r)
    precision.append(p)
    TPR.append(tpr)
    FPR.append(fpr)

    if len(FPR) > 1:
        auc += (FPR[-2] - FPR[-1]) * (TPR[-1] + TPR[-2])

# 绘制ROC曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('ROC curve', fontsize=14)
plt.plot(FPR, TPR, marker='o')
plt.ylabel('TPR', fontsize=16)
plt.xlabel('FPR', fontsize=16)

# 绘制Precision-Recall曲线
plt.subplot(1, 2, 2)
plt.title('Precision-Recall curve', fontsize=14)
plt.plot(recall, precision, marker='o')
plt.ylabel('Precision', fontsize=16)
plt.xlabel('Recall', fontsize=16)

# 打印AUC值
print('AUC=%.2f' % (auc / 2))

plt.show()
