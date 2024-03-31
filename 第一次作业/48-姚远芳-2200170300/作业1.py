# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:19:36 2024

@author: Administrator
"""

import matplotlib.pyplot as plt

true = [[0,0,1],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0]]
predict = [[0.1,0.2,0.7],[0.1,0.6,0.3],[0.5,0.2,0.3],[0.1,0.1,0.8],[0.4,0.2,0.4],[0.6,0.3,0.1],[0.4,0.2,0.4],[0.4,0.1,0.5],[0.1,0.1,0.8],[0.1,0.8,0.1]]
avergex=[0.0,0.0,0.0]
avergey=[0.0,0.0,0.0]

for i in range(3):
    TPR=[]#真实利率
    FPR=[]#假正例率
    
    p1 = [item[i] for item in true]
    aa = [item[i] for item in predict]
    
    pp = list(zip(p1, aa))
    pp.sort(key=lambda x: x[1], reverse=True)
    
    aa.sort(reverse=True)
    
    for a in aa:
        tp = 0
        fn = 0
        fp = 0
        tn = 0
 
        for p in pp:
            if p[0] == 1 and p[1] >= a:
                tp += 1
            elif p[0] == 1 and p[1] < a:
                fn += 1
            elif p[0] == 0 and p[1] >= a:
                fp += 1
            elif p[0] == 0 and p[1] < a:
                tn += 1
            
        x = tp / (tp + fn) if tp + fn != 0 else 0
        fpr = fp / (tn + fp) if tn + fp != 0 else 0
    
        TPR.append(x)
        FPR.append(fpr)
    
    avergex = [x+y for x,y in zip(avergex, TPR)]
    avergey = [x+y for x,y in zip(avergey, FPR)]
    
    plt.figure(i+1)
    plt.figure(figsize=(5,5))
    plt.title('roc curve-' + str(i+1), fontsize=14)
    plt.plot(FPR, TPR)
    plt.plot(FPR, TPR, 'ro')
    plt.ylabel('TPR-' + str(i+1), fontsize=16)
    plt.xlabel('FPR-' + str(i+1), fontsize=16)

avergex = [round(x/3, 4) for x in avergex]
avergey = [round(y/3, 4) for y in avergey]

plt.figure(figsize=(5,5))
plt.title('roc-average', fontsize=14)
plt.xlabel('FPR-AVERAGE', fontsize=16)
plt.ylabel('TPR-AVERAGE', fontsize=16)
plt.plot(avergey, avergex)
plt.plot(avergey, avergex, 'go')
plt.show()
