import matplotlib.pyplot as plt
import numpy as np
pp = [['1', 0.9], ['1', 0.7], ['0', 0.65], ['1', 0.6], ['0', 0.5], ['0', 0.4], ['0', 0.4], ['1', 0.4], ['0', 0.35],
      ['0', 0.2]]
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
        if (p[0] == '1') and (p[1] >= a):
            tp += 1
        elif (p[0] == '1') and (p[1] < a):
            fn += 1
        elif (p[0] == '0') and (p[1] >= a):
            fp += 1
        elif (p[0] == '0') and (p[1] < a):
            tn += 1

    if tp + fn != 0:
        x = float(tp) / (tp + fn)
    else:
        x = 0

    if tp + fp != 0:
        y = float(tp) / (tp + fp)
    else:
        y = 0

    if tn + fp != 0:
        fpr = float(fp) / (tn + fp)
    else:
        fpr = 0

    recall.append(x)
    precision.append(y)
    TPR.append(x)
    FPR.append(fpr)

plt.figure(figsize=(5, 5))
plt.title('Precision-Recall Curve', fontsize=16)
plt.plot(recall, precision)
plt.plot(recall, precision, 'ro')
plt.ylabel('Precision', fontsize=16)
plt.xlabel('Recall', fontsize=16)

plt.figure(num=2, figsize=(5, 5))
plt.title('ROC curve', fontsize=14)
plt.plot(FPR, TPR)
#画出这些点连成的线
plt.plot(FPR, TPR, 'ro') # 画出点，并标红
plt.ylabel('TPR',fontsize =14)
plt.xlabel ('FPR',fontsize =14)
auc = np.trapz(TPR, FPR)
print('auc=%.2f' % auc)

plt.show()
