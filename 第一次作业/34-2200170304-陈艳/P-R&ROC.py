# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:32:54 2024

@author: lenovo
"""
import matplotlib.pyplot as plt
import numpy as np
# pp = [['T,0.9'],['T',0.8],['N',0.7],['T',0.6],['T',0.55],
#       ['T',0.54],['N',0.53],['N',0.52],['T',0.51],['N',0.505],
#       ['T',0.4],['T',0.39],['T',0.38],['N',0.37],['N',0.36],
#       ['N',0.35],['T',0.34],['N',0.33],['T',0.30],['N',0.1]]

pp = [['1',0.9],['0',0.4],['0',0.2],['1',0.6],['0',0.5],
      ['0',0.4],['1',0.7],['1',0.4],['0',0.65],['0',0.35]]

aa = [0.9,0.4,0.2,0.6,0.5,0.4,0.7,0.4,0.65,0.35]

recall =[]
precision =[]
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
        if(p[0] =='1') and (p[1]>=a):
            tp = tp+1
        elif(p[0] =='1') and (p[1]<a):
            fn = fn+1
        elif(p[0] =='0') and (p[1]>=a):
            fp = fp+1
        elif(p[0] =='0') and (p[1]<a):
            tn = tn+1
            
    x = float(tp)/(tp+fn)
    if tp + fp == 0:
        continue
    y = float(tp)/(tp+fp)
    fpr = float(fp)/(tn+fp)
    
    recall.append(x)
    precision.append(y)
    TPR.append(x)
    FPR.append(fpr)
#绘制p-r曲线
# recall.sort()  
# plt.figure(figsize =(5,5))
# plt.title('P-R curve',fontsize = 16)
# plt.plot(recall,precision)
# plt.plot(recall,precision,'ro')
# plt.ylabel('Precision',fontsize = 16)
# plt.xlabel('recall',fontsize = 16)

#绘制ROC曲线
FPR.sort()  
plt.figure(figsize =(5,5))
plt.title('ROC curve',fontsize = 16)
plt.plot(FPR,TPR)
plt.plot(FPR,TPR,'ro')
plt.ylabel('TPR',fontsize = 16)
plt.xlabel('FPR',fontsize = 16)

#计算AUC
i=0
auc = 0
while(i<9):
    auc = auc + (FPR[i+1]-FPR[i])*(TPR[i]+TPR[i+1])
    i=i+1

auc = float(auc/2)
print ('auc=%.2f' % auc)
    
    
    
    
    
    
    
    

