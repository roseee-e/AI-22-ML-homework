import matplotlib.pyplot as plt
import numpy as np

pp = [['T',0.9],['T',0.7],['F',0.65],['T',0.6],['F',0.5],['F',0.4],['F',0.4],['T',0.4],['F',0.35],['F',0.2]]
aa = [0.9,0.7,0.65,0.6,0.5,0.4,0.4,0.4,0.35,0.2]
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
            tp += 1
        elif (p[0] == 'T') and (p[1] < a):
            fn += 1
        elif (p[0] == 'F') and (p[1] >= a):
            fp += 1
        elif (p[0] == 'F') and (p[1] < a):
            tn += 1

    x = float(tp) / (tp + fn)
    y = float(tp) / (tp + fp)
    fpr = float(fp) / (fp + tn)

    recall.append(x)
    precision.append(y)
    TPR.append(x)
    FPR.append(fpr)

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.title('P-R Graph')
plt.plot(recall, precision)
plt.plot(recall, precision, 'ro')
plt.ylabel('Precision')
plt.xlabel('Recall')

plt.subplot(2, 1, 2)
plt.title('ROC Graph')
plt.plot(FPR, TPR)
plt.plot(FPR, TPR, 'ro')
plt.ylabel('TPR')
plt.xlabel('FPR')

plt.tight_layout()
plt.show()