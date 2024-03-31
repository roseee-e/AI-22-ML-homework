# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 22:33:58 2024

@author: Lenovo
"""

import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import metrics  
from sklearn.metrics import roc_curve, auc  
true_labels = np.array([[0, 0, 1],[0, 1, 0],[1, 0, 0],[0, 0, 1],[1, 0, 0],[0, 1, 0],[0, 1, 0],[0, 1, 0],[0, 0, 1],[0, 1, 0]])  
predict_scores = np.array([[0.1, 0.2, 0.7],[0.1, 0.6, 0.3],[0.5, 0.2, 0.3],[0.1, 0.1, 0.8],[0.4, 0.2, 0.4],[0.6, 0.3, 0.1],[0.4, 0.2, 0.4],[0.4, 0.1, 0.5],[0.1, 0.1, 0.8],[0.1, 0.8, 0.1]])  
fpr_micro, tpr_micro, thresholds_micro = roc_curve(true_labels.ravel(), predict_scores.ravel())  
roc_auc_micro = auc(fpr_micro, tpr_micro)  
colors = ['green', 'orange', 'blue']  
fpr = dict()  
tpr = dict()  
roc_auc = dict()  
  

for i in range(true_labels.shape[1]):  
    fpr[i], tpr[i], thresholds = roc_curve(true_labels[:, i], predict_scores[:, i])  
    roc_auc[i] = auc(fpr[i], tpr[i])  
plt.figure()  
for i in range(true_labels.shape[1]):  
    plt.plot(fpr[i], tpr[i], color=colors[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')  
plt.plot(fpr_micro, tpr_micro, color='red', linestyle='--',  
         label='Micro-average ROC(AUC = {0:0.2f})'.format(roc_auc_micro))  
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')  
plt.xlabel('FPR')  
plt.ylabel('TPR')  
plt.legend(loc="lower right")  
plt.show()