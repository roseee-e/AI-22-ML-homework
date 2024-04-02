import sys
sys.path.append("D:\python\lib\site-packages")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve,auc

pp = [['T', 0.9], ['N', 0.4], ['N', 0.2], ['T', 0.6], ['N', 0.5], ['N', 0.4], ['T', 0.7], ['T', 0.4], ['N', 0.65],
      ['N', 0.35]]
y_true=[]
y_score=[]
for p in pp:
    y_c=p[0]
    if y_c=='T':
        y=1
    else:
        y=0

    y_hat=p[1]
    y_true.append(y)
    y_score.append(y_hat)
precision,recall,_=precision_recall_curve(y_true,y_score)
roc_auc=roc_auc_score(y_true,y_score,sample_weight=None )
pr_auc=auc(recall,precision )
print('pr_auc=%.2f' % pr_auc)
print('roc_auc=%.2f' % roc_auc)
pp.sort(key=lambda x: x[1], reverse=True)
aa = [0.9, 0.4, 0.2, 0.6, 0.5, 0.4, 0.7, 0.4, 0.65, 0.35]
aa.sort(reverse=True)
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
        if (p[0] == 'T') and (p[1] >= a):
            tp = tp + 1
        elif (p[0] == 'T') and (p[1] < a):
            fn = fn + 1
        elif (p[0] == 'N') and (p[1] >= a):
            fp = fp + 1
        elif (p[0] == 'N') and (p[1] < a):
            tn = tn + 1

    x = float(tp) / (tp + fn)
    y = float(tp) / (tp + fp)
    if (tn + fp) == 0:  # 注意除数不能为零
        fpr = 0
    else:
        fpr = float(fp) / (tn + fp)

    recall.append(x)
    precision.append(y)
    TPR.append(x)
    FPR.append(fpr)

plt.figure(1)  # 分为两个图画出图像
plt.figure(figsize=(5, 5))
plt.title('roc curve', fontsize=14)
plt.plot(FPR, TPR)
plt.plot(FPR, TPR, 'ro')
plt.ylabel('TPR', fontsize=16)
plt.xlabel('FPR', fontsize=16)
plt.figure(2)
plt.figure(figsize=(5, 5))
plt.title('precision-recall curve', fontsize=16)
plt.plot(recall, precision)
plt.plot(recall, precision, 'ro')
plt.ylabel('Precision', fontsize=16)
plt.xlabel('Recall', fontsize=16)
plt.show()
