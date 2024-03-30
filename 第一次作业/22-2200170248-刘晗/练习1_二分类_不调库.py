import numpy as np
import matplotlib.pyplot as plt

# 真实标签
y_true1 = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])
# 预测概率
y_score1 = np.array([0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35])

# 根据预测概率的降序排列索引
index = np.argsort(y_score1)[::-1]
# 根据排序后的索引重新排列 y_true 和 y_score
y_true = y_true1[index]
y_score = y_score1[index]

precision = []
recall = TPR = []
FPR = []

for i in y_score:

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for k in range(len(y_true)):
        if (y_true[k] == 1) and (y_score[k] >= i):
            tp = tp + 1
        elif (y_true[k] == 1) and (y_score[k] < i):
            fn = fn + 1
        elif (y_true[k] == 0) and (y_score[k] >= i):
            fp = fp + 1
        elif (y_true[k] == 0) and (y_score[k] < i):
            tn = tn + 1
    x = float(tp) / (tp + fn)
    y = float(tp) / (tp + fp)
    fpr = float(fp) / (fp + tn)

    recall.append(x)
    precision.append(y)
    FPR.append(fpr)

# 一个窗口中划分多个绘图区域  plt.subplots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# 画PR曲线

axs[0].plot(recall, precision)
axs[0].plot(recall, precision, 'ro')
axs[0].set_title('PR Curve', fontsize=14)
axs[0].set_xlabel('Recall', fontsize=14)
axs[0].set_ylabel('Precision', fontsize=14)

# 计算 AUC
auc = 0.0
for i in range(1, len(FPR)):
    auc += (FPR[i] - FPR[i - 1]) * (TPR[i] + TPR[i - 1]) / 2
print("AUC的值为{}".format(auc))

# 画ROC曲线
axs[1].plot(FPR, TPR)
axs[1].plot(FPR, TPR, 'ro')  # 画出点，并标红
axs[1].set_title("ROC curve", fontsize=14)
axs[1].set_ylabel("TPR", fontsize=14)
axs[1].set_xlabel("FPR", fontsize=14)
plt.show()
