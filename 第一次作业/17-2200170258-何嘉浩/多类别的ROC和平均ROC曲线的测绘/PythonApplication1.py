import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import auc
data_true=np.asarray([[0,0,1],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0]])
data_pre=np.asarray([[0.1,0.2,0.7],[0.1,0.6,0.3],[0.5,0.2,0.3],[0.1,0.1,0.8],[0.4,0.2,0.4],[0.6,0.3,0.1],[0.4,0.2,0.4],[0.4,0.1,0.5],[0.1,0.1,0.8],[0.1,0.8,0.1]])
print(data_true.shape,data_pre.shape)

data_classes=len(data_true[1,:])
fpr=dict()
tpr=dict()
thre=dict()
tpr_micro=dict()
fpr_micro=dict()
roc_auc=dict()
for i in range(data_classes):
    fpr[i],tpr[i],thre[i]=roc_curve(data_true[:,i],data_pre[:,i])
    plt.figure(figsize=(5, 5))  
    plt.title('ROC Curve({})'.format(i),fontsize=16)  
    plt.plot(fpr[i], tpr[i])  
    plt.plot(fpr[i], tpr[i],'ro')  
    plt.xlabel('tpr[{}]'.format(i),fontsize=16)  
    plt.ylabel('fpr[{}]'.format(i),fontsize=16)  
    plt.show()

fpr_micro, tpr_micro, _ = roc_curve(data_true.ravel(), data_pre.ravel())
plt.figure(figsize=(5, 5))  
plt.title('ROC Curve(micro)',fontsize=16) 
plt.plot(fpr_micro, tpr_micro) 
plt.plot(fpr_micro, tpr_micro,'ro') 
plt.xlabel('tpr[micro]',fontsize=16)  
plt.ylabel('fpr[micro]',fontsize=16)  
plt.show()
