import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve,auc

Y_true=[1,0,0,1,0,0,1,1,0,0]
Y_score=[0.9,0.4,0.2,0.6,0.5,0.4,0.7,0.4,0.65,0.35]

pre,rec,thre=precision_recall_curve(Y_true,Y_score)
pr_auc=auc(rec,pre)
plt.figure(figsize=(5, 5))  
plt.title('Precision-Recall Curve',fontsize=16)  
plt.plot(rec, pre)  
plt.plot(rec, pre,'ro')  
plt.xlabel('Recall',fontsize=16)  
plt.ylabel('Precision',fontsize=16)  
plt.show()
print(pr_auc)


fpr, tpr, roc_thre = roc_curve(Y_true, Y_score)  
roc_auc = auc(fpr, tpr)  
plt.figure(figsize=(5, 5))  
plt.title('ROC Curve',fontsize=16)  
plt.plot(fpr, tpr)  
plt.plot(fpr, tpr,'ro')  
plt.xlabel('tpr',fontsize=16)  
plt.ylabel('fpr',fontsize=16)  
plt.show()
print(roc_auc)
