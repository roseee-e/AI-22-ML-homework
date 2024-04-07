import matplotlib.pyplot as plt
import numpy as np
#首先对得到的数据进行降序排序
data = [['T',0.9],['N',0.4],['N',0.2],['T',0.6],['N',0.5],['N',0.4],['T',0.7],['T',0.4],['N',0.65],['N',0.35]]
data.sort(key=lambda x: x[1], reverse=True)
thresholds = [0.9, 0.4, 0.2, 0.6, 0.5, 0.4, 0.7, 0.4, 0.65, 0.35]
thresholds.sort(reverse=True)
recall = []
precision = []
TPR = []
FPR = []
for threshold in thresholds:
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for point in data:
        if (point[0] == 'T') and (point[1] >= threshold):
            tp += 1
        elif (point[0] == 'T') and (point[1] < threshold):
            fn += 1
        elif (point[0] == 'N') and (point[1] >= threshold):
            fp += 1
        elif (point[0] == 'N') and (point[1] < threshold):
            tn += 1
    recall.append(float(tp) / (tp + fn))
    precision.append(float(tp) / (tp + fp))
    TPR.append(float(tp) / (tp + fn))
    FPR.append(0 if (tn + fp) == 0 else float(fp) / (tn + fp))
plt.figure(1)
plt.figure(figsize=(5, 5))
plt.title('ROC Curve', fontsize=14)
plt.plot(FPR, TPR)
plt.plot(FPR, TPR, 'ro')
plt.ylabel('TPR', fontsize=16)
plt.xlabel('FPR', fontsize=16)
plt.figure(2)
plt.figure(figsize=(5, 5))
plt.title('Precision-Recall Curve', fontsize=16)
plt.plot(recall, precision)
plt.plot(recall, precision, 'ro')
plt.ylabel('Precision', fontsize=16)
plt.xlabel('Recall', fontsize=16)
plt.show()
i = 0
auc = 0
while (i < 9):
    auc += (FPR[i+1] - FPR[i]) * (TPR[i] + TPR[i+1])
    i += 1
auc /= 2
print('AUC=%.2f' % auc)
