import matplotlib.pyplot as plt
import numpy as np

pp = [['T', 0.90], ['T', 0.70], ['N', 0.65], ['T', 0.60], ['N', 0.50], ['N', 0.40], ['N', 0.40], ['T', 0.40],
      ['N', 0.35], ['N', 0.20]]

aa = [0.90, 0.70, 0.65, 0.60, 0.50, 0.40, 0.40, 0.40, 0.35, 0.20]

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
        if (p[0] == 'T') and (p[1] >= a):
            tp = tp + 1
        elif (p[0] == 'T') and (p[1] < a):
            fn = fn + 1
        elif (p[0] == 'N') and (p[1] >= a):
            fp = fp + 1
        elif (p[0] == 'N') and (p[1] < a):
            tn = tn + 1

    x = float(tp) / (tp + fn)
    y = float(tp) / (tp + fp)
    fpr = float(fp) / (tn + fp)

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
plt.legend(["P-R"],loc='upper right')

plt.show()
