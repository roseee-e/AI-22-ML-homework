import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

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

AUC = roc_auc_score(Y_ture, Y_score)

fpr, tpr, th = roc_curve(Y_ture, Y_score)

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr)
plt.plot(fpr, tpr, 'ro')
plt.xlabel('FPR', fontsize=16)
plt.ylabel('TPR', fontsize=16)
plt.title('ROC curve', fontsize=16)

print("AUC=", AUC)
plt.show()
# 图像绘制结果与AUC的值与不调用sklearn时一样