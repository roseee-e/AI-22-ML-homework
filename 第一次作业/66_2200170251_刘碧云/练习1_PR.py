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

for a in aa:   # 遍历数组
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    x = 0
    y = 0

    for p in pp:   # 遍历list
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
plt.title('precision-recall curve', fontsize=14)
plt.plot(recall, precision)
plt.plot(recall, precision, 'ro')
plt.ylabel('Precision', fontsize=14)
plt.xlabel('Recall', fontsize=14)

plt.grid(True)  # 显示网格线

plt.show()

