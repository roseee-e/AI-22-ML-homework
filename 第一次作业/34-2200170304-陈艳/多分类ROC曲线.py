# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:42:54 2024

@author: lenovo
"""

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 输入数据
y_true = np.asarray([[0,0,1],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,1,0],
                     [0,1,0],[0,1,0],[0,0,1],[0,1,0]])
y_pred = np.asarray([[0.1,0.2,0.7],[0.1,0.6,0.3],[0.5,0.2,0.3],
                     [0.1,0.1,0.8],[0.4,0.2,0.4],[0.6,0.3,0.1],
                     [0.4,0.2,0.4],[0.4,0.1,0.5],[0.1,0.1,0.8],[0.1,0.8,0.1]])

# 计算每个类别的TPR和FPR，并绘制ROC曲线
n_classes = y_true.shape[1]
plt.figure()

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
    

    
    plt.plot(fpr, tpr, label='curve {}')
    
# 绘制平均ROC曲线（micro average）
fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_pred.ravel())


plt.plot(fpr_micro, tpr_micro, label='micro-average', linestyle=':', linewidth=2)


# 设置图例和标题
plt.legend(loc='lower right')
plt.title('ROC curve')
plt.xlabel('FPR')
plt.ylabel('TPR')

# 显示图像
plt.show()
