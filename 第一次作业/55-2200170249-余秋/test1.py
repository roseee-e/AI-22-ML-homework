import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])
y = np.array([0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35])

# 根据预测概率的降序排列索引
index = np.argsort(x)[::-1]
y_true = x[index]
y_score = y[index]

precision = []
recall = []
TPR = []
FPR = []

for i in y_score:

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for k in range(len(y_true)):
        if y_true[k] == 1 and y_score[k] >= i:
            tp = tp + 1
        elif y_true[k] == 1 and y_score[k] < i:
            fn = fn + 1
        elif y_true[k] == 0 and y_score[k] >= i:
            fp = fp + 1
        elif y_true[k] == 0 and y_score[k] < i:
            tn = tn + 1
    x = float(tp) / (tp + fn)
    y = float(tp) / (tp + fp)
    fpr = float(fp) / (fp + tn)

    recall.append(x)
    TPR.append(x)
    precision.append(y)
    FPR.append(fpr)

# 画PR曲线

plt.plot(recall, precision)
plt.plot(recall, precision)
plt.set_title('PR-Curve')
plt.set_xlabel('Recall')
plt.set_ylabel('Precision')

# 计算AUC
auc = 0.0
for i in range(1, len(FPR)):
    auc += (FPR[i] - FPR[i - 1]) * (TPR[i] + TPR[i - 1]) / 2

# ROC曲线
plt.plot(FPR, TPR)
plt.set_title("ROC-curve")
plt.set_ylabel("TPR")
plt.set_xlabel("FPR")
plt.show()
