import matplotlib.pyplot as plt
import numpy as np

pp = [[1, 0.90], [0, 0.40], [0, 0.20], [1, 0.60], [0, 0.50], [0, 0.40]
    , [1, 0.70], [1, 0.40], [0, 0.65], [0, 0.35]]
# 构建样本的list类型数据

pp.sort(key=lambda x: x[1], reverse=True)
# 对数据集按照score降序

Y_ture = [x[0] for x in pp]
# 其中1代表恶性，0代表良性
Y_score = [x[1] for x in pp]
# Y_score代表机器学习方法预测结果（即为恶性的概率，恶性代表正例）

w = np.linspace(0.35, 0.9, num=1000)
# 用等长的阈值范围w

precision = []
recall = []
TPR = []
FPR = []

for a in Y_score:
    # 把Y_score换成w，图像结果一样
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    x = 0
    y = 0

    for p in pp:
        if (p[0] == 1) and (p[1] >= a):
            tp += 1
        elif (p[0] == 1) and (p[1] < a):
            fn += 1
        elif (p[0] == 0) and (p[1] >= a):
            fp += 1
        elif (p[0] == 0) and (p[1] < a):
            tn += 1

    x = float(tp) / (tp + fn)
    y = float(tp) / (tp + fp)
    fpr = float(fp) / (fp + tn)

    recall.append(x)
    precision.append(y)
    FPR.append(fpr)
    TPR.append(x)

plt.figure(figsize=(5, 5))
plt.plot(recall, precision)
plt.plot(recall, precision, 'ro')
plt.xlabel('Recall', fontsize=16)
plt.ylabel('Precision', fontsize=16)
plt.title('precision-recall curve', fontsize=16)
# 绘制PR曲线

plt.figure(figsize=(5, 5))
plt.plot(FPR, TPR)
plt.plot(FPR, TPR, 'ro')
plt.xlabel('FPR', fontsize=16)
plt.ylabel('TPR', fontsize=16)
plt.title('ROC curve', fontsize=16)
# 绘制ROC曲线

AUC = 0
i = 0
while (i < 9):
    AUC += (FPR[i + 1] - FPR[i]) * (TPR[i] + TPR[i + 1])
    i += 1
AUC = float(AUC) / 2
# 计算AUC

print("AUC=", AUC)
plt.show()
