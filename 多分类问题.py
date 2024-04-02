# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:18:42 2024

@author: 86182
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# 第一条曲线
you_true = [[0,0,1],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0]]
you_pred = [[0.1,0.2,0.7],[0.1,0.6,0.3],[0.5,0.2,0.3],[0.1,0.1,0.8],[0.4,0.2,0.4],[0.6,0.3,0.1],
            [0.4,0.2,0.4],[0.4,0.1,0.5],[0.1,0.1,0.8],[0.1,0.8,0.1]]
pp1 = [[0, 0.1], [0, 0.1], [1, 0.5], [0, 0.1], [1, 0.4], [0, 0.6], [0, 0.4], [0, 0.4], [0, 0.1], [0, 0.1]]
pp1.sort(key=lambda x: x[1])
pp = pp1[::-1]
you_true1 = [i[0] for i in pp]
you_score1 = [i[1] for i in pp]
fpr1, tpr1, _ = roc_curve(you_true1, you_score1)

# 第二条曲线
pp1 = [[0, 0.2], [1, 0.6], [0, 0.2], [0, 0.1], [0, 0.2], [1, 0.3], [1, 0.2], [1, 0.1], [0, 0.1], [1, 0.8]]
pp1.sort(key=lambda x: x[1])
pp = pp1[::-1]
you_true2 = [i[0] for i in pp]
you_score2 = [i[1] for i in pp]
fpr2, tpr2, _ = roc_curve(you_true2, you_score2)

# 第三条曲线
pp1 = [[1, 0.7], [0, 0.3], [0, 0.3], [1, 0.8], [0, 0.4], [0, 0.1], [0, 0.4], [0, 0.5], [1, 0.8], [0, 0.1]]
pp1.sort(key=lambda x: x[1])
pp = pp1[::-1]
you_true3 = [i[0] for i in pp]
you_score3 = [i[1] for i in pp]
fpr3, tpr3, _ = roc_curve(you_true3, you_score3)

# 绘制前三条曲线
plt.plot(fpr1, tpr1, label='Curve 1')
plt.plot(fpr2, tpr2, label='Curve 2')
plt.plot(fpr3, tpr3, label='Curve 3')

# 计算第四条曲线的FPR和TPR的平均值
fpr4 = [(fpr1[i] + fpr2[i] + fpr3[i]) / 3 for i in range(len(fpr1))]
tpr4 = [(tpr1[i] + tpr2[i] + tpr3[i]) / 3 for i in range(len(tpr1))]

# 绘制第四条曲线
plt.plot(fpr4, tpr4, label='Average Curve', linestyle='--')

# 设置图表标题和轴标签
plt.title("ROC Curve", fontsize=14)
plt.ylabel("True Positive Rate (TPR)", fontsize=14)
plt.xlabel("False Positive Rate (FPR)", fontsize=14)

plt.legend()

plt.show()






















