import matplotlib.pyplot as plt
import numpy as np

pp = [['T', 0.9], ['T', 0.7], ['F', 0.65], ['T', 0.6], ['F', 0.5], ['F', 0.4],
      ['F', 0.4], ['T', 0.4], ['F', 0.35], ['F', 0.2]]
aa = [0.9, 0.7, 0.65, 0.6, 0.5, 0.4, 0.4, 0.4, 0.35, 0.2]
recall = []
precision = []
TPR = []
FPR = []

for a in aa:
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for p in pp:
        if (p[0] == 'T') and (p[1] >= a):
            tp = tp+1
        if (p[0] == 'T') and (p[1] < a):
            fn = fn+1
        if (p[0] == 'F') and (p[1] >= a):
            fp = fp+1
        if (p[0] == 'F') and (p[1] < a):
            tn = tn+1
    x = float(tp)/(tp+fn)
    y = float(tp)/(tp+fp)
    fpr = float(fp)/(tn+fp)
    recall.append(x)
    precision.append(y)
    TPR.append(x)
    FPR.append(fpr)

plt.figure(figsize=(5, 5))
plt.title('precision-recall curve', fontsize=16)
plt.plot(recall, precision)
plt.plot(recall, precision, 'ro')
plt.ylabel('Precision', fontsize=16)
plt.xlabel('Recall', fontsize=16)
plt.show()
plt.figure(figsize=(5, 5))
plt.title('ROC curve', fontsize=14)
plt.plot(FPR, TPR)
plt.plot(FPR, TPR, 'ro')
plt.ylabel('TPR', fontsize=14)
plt.xlabel('FPR', fontsize=14)
plt.show()
# 计算auc
auc = 0
for i in range(9):
    auc = auc+(FPR[i+1]-FPR[i])*(TPR[i+1]+TPR[i])

print('auc=%.2f' % auc)
