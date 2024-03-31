# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 19:54:27 2024

@author: 86182
"""

import matplotlib.pyplot as plt
import numpy as np

pp1 = [[1,0.90],[0,0.40],[0,0.20],[1,0.60],[0,0.50],[0,0.40],[1,0.70],[1,0.40],[0,0.65],[0,0.35]]
#pp = pp1.sort(key=lambda )
pp1.sort(key=lambda pp1:pp1[1])
pp = pp1[::-1]

aa1 = [0.90, 0.40, 0.20, 0.60,0.50,0.40,0.70,0.40,0.65,0.35]
aa1.sort()
aa = aa1[::-1]


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
        if (p[0] == 1) and (p[1] >= a):
            tp+=1
        elif (p[0] == 1) and (p[1] < a):
            fn+=1
        elif (p[0] == 0) and (p[1] >= a):
            fp+=1
        elif (p[0] == 0) and (p[1] < a):
            tn+=1
    x = float(tp)/(tp + fn)
    y = float(tp)/(tp + fp)
    fpr = float(fp)/(tn + fp)
    
    recall.append(x)
    precision.append(y)
    TPR.append(x)
    FPR.append(fpr)

#绘制PR曲线
recall.sort()
plt.figure()
plt.title('precision-recall curve',fontsize=20)
plt.plot(recall,precision)
plt.plot(recall,precision,'ro')
plt.ylabel('Precision',fontsize=20)
plt.xlabel('Recall',fontsize=20)
    

# 绘制ROC曲线
FPR.sort()
plt.figure(figsize=(5,5))
plt.title('ROC curve',fontsize=14)
plt.plot(FPR,TPR)
plt.plot(FPR,TPR,'ro')
plt.ylabel('TPR',fontsize=14)
plt.xlabel('FPR',fontsize=14)


# 求AUC的值
i = 0
auc = 0
while(i < 8):
    auc = auc + (FPR[i+1] - FPR[i]) * (TPR[i] + TPR[i+1])
    i+=1
auc = float(auc/2)
print('auc=%.2f'%auc)














