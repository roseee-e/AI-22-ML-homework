import matplotlib.pyplot as plt

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

# 计算AUC
i = 0
auc = 0
while i < 9:
    auc = auc + (FPR[i + 1] - FPR[i]) * (TPR[i] + TPR[i + 1])
    i = i + 1
auc = float(auc / 2)
print('该ROC曲线对应的AUC=%.2f' % auc)

plt.figure(figsize=(6, 6))
plt.title('ROC curve', fontsize=14)
plt.plot(FPR, TPR)
plt.plot(FPR, TPR, 'ro')
plt.ylabel('TPR', fontsize=14)
plt.xlabel('FPR', fontsize=14)
plt.legend(["ROC curve (AUC = 0.83)"])

plt.show()
