# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:13:07 2024

@author: 86182
"""
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_curve, auc  
  
# 给定的真实标签和预测分数  
true_labels = np.array([[0, 0, 1],  
                         [0, 1, 0],  
                         [1, 0, 0],  
                         [0, 0, 1],  
                         [1, 0, 0],  
                         [0, 1, 0],  
                         [0, 1, 0],  
                         [0, 1, 0],  
                         [0, 0, 1],  
                         [0, 1, 0]])  
predict_scores = np.array([[0.1, 0.2, 0.7],  
                            [0.1, 0.6, 0.3],  
                            [0.5, 0.2, 0.3],  
                            [0.1, 0.1, 0.8],  
                            [0.4, 0.2, 0.4],  
                            [0.6, 0.3, 0.1],  
                            [0.4, 0.2, 0.4],  
                            [0.4, 0.1, 0.5],  
                            [0.1, 0.1, 0.8],  
                            [0.1, 0.8, 0.1]])  
  
# 初始化用于存储每个类别ROC曲线数据的列表  
fpr = {}  
tpr = {}  
roc_auc = {}  
  
# 计算每个类别的ROC曲线  
num_classes = 3  
for i in range(num_classes):  
    fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predict_scores[:, i])  
    roc_auc[i] = auc(fpr[i], tpr[i])  
  
# 绘制每个类别的ROC曲线  
plt.figure()  
colors = ['aquamarine', 'darkorange', 'cornflowerblue']  
for i in range(num_classes):  
    plt.plot(fpr[i], tpr[i], color=colors[i],  
             label=f'Class {i} (AUC = {roc_auc[i]:.2f})')  
  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Receiver Operating Characteristic Curves for Each Class')  
  
# 计算micro-average ROC曲线  
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))  
micro_tpr = np.zeros_like(all_fpr)  
for i in range(num_classes):  
    micro_tpr += np.interp(all_fpr, fpr[i], tpr[i])  
micro_tpr /= num_classes  
micro_auc = auc(all_fpr, micro_tpr)  
  
# 绘制micro-average ROC曲线  
plt.plot(all_fpr, micro_tpr,  
         color='black', linestyle=':', linewidth=2,  
         label='micro-average ROC curve (AUC = {0:0.2f})'.format(micro_auc))  
  
# 显示图例  
plt.legend(loc="lower right")  
plt.show()
