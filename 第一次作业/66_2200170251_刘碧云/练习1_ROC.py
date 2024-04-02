import matplotlib.pyplot as plt
import numpy as np

# 构建lis类型数据
pp = [['T', 9], ['T', 0.7], ['N', 0.65], ['T', 0.6], ['N', 0.5], ['N', 0.4], ['N', 0.4], ['T', 0.4],
      ['N', 0.35], ['N', 0.2]]
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
plt.title('ROC curve', fontsize=14)
plt.plot(FPR, TPR)
plt.plot(FPR, TPR, 'ro')
plt.ylabel('TPR', fontsize=14)
plt.xlabel('FPR', fontsize=14)

plt.grid(True)  # 显示网格线

# 计算AUC
i = 0
auc = 0
while(i < 9):
    auc = auc + (FPR[i + 1] - FPR[i]) * (TPR[i] + TPR[i + 1])
    i = i+1

auc = float(auc / 2)

print('auc = %.2f' % auc)

plt.show()
