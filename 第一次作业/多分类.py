import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve

y_true=np.asarray([[0,0,1],[0,1,0],
        [1,0,0],[0,0,1],
        [1,0,0],[0,1,0],
        [0,1,0],[0,1,0],
        [0,0,1],[0,1,0]
        ])
y_pre=np.asarray([[0.1,0.2,0.7],[0.1,0.6,0.3],
       [0.5,0.2,0.3],[0.1,0.1,0.8],
       [0.4,0.2,0.4],[0.6,0.3,0.1],
       [0.4,0.2,0.4],[0.4,0.1,0.5],
       [0.1,0.1,0.8],[0.1,0.8,0.1]
       ])

n_class=len(y_true[1,:])
#print(n_class)
fpr=dict()
tpr=dict()
roc_auc=dict()
for i in range(n_class):
    fpr[i],tpr[i],_=roc_curve(y_true[:,i],y_pre[:,i])
    roc_auc[i]=roc_auc_score(y_true[:,i],y_pre[:,i],sample_weight=None)

#micro的指标
#fpr['micro'],tpr['micro'],_=roc_curve(y_true.ravel(),y_pre.ravel())

#macro的指标
fpr_grid=np.linspace(0.0,1.0,100)
mean_tpr=np.zeros_like(fpr_grid)
for i in range(n_class):
    mean_tpr+=np.interp(fpr_grid,fpr[i],tpr[i])

mean_tpr/=n_class
fpr['macro']=fpr_grid
tpr['macro']=mean_tpr


#weight的指标
'''fpr_grid=np.linspace(0.0,1.0,100)
y_true_list=list([tuple(t) for t in y_true])
classNum=dict((a,y_true_list.count(a)) for a in y_true_list)
n1=classNum[(1,0,0)]
n2=classNum[(0,1,0)]
n3=classNum[(0,0,1)]
ratio=[n1/(n1+n2+n3),n2/(n1+n2+n3),n3/(n1+n2+n3)]
avg_tpr=np.zeros_like(fpr_grid)
for i in range(n_class):
    avg_tpr+=ratio[i]*np.interp(fpr_grid,fpr[i],tpr[i])

fpr['weight']=fpr_grid
tpr['weight']=avg_tpr
'''


for i in range(n_class):
    plt.plot(fpr[i], tpr[i],label='Roc'+str(i))
    plt.title('ROC curve', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.xlabel('FPR', fontsize=14)
#plt.plot(fpr['micro'],tpr['micro'],label='Roc_micro')
plt.plot(fpr['macro'],tpr['macro'],label='Roc_macro')
#plt.plot(fpr['weight'],tpr['weight'],label='Roc_weight')
plt.legend(loc='upper right')
plt.show()