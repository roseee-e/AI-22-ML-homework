import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

y_true=np.asarray([[0,0,1],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0]])
y_pred=np.asarray([[0.1,0.2,0.7],[0.1,0.6,0.3],[0.5,0.2,0.3],[0.1,0.1,0.8],[0.4,0.2,0.4],[0.6,0.3,0.1],[0.4,0.2,0.4],[0.4,0.1,0.5],[0.1,0.1,0.8],[0.1,0.8,0.1],])
print(y_true.shape,y_pred.shape)

n_classes=len(y_true[1,:1])
fpr =dict()
tpr =dict()
roc_auc=dict()

for i in range(n_classes):
    fpr[i],tpr[i],_=roc_curve(y_true[:,i],y_pred[:,i])
    roc_auc[i]=auc(fpr[i],tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())

fpr_grid=np.linspace(0.0,1.0,100)
mean_tpr=np.zeros_like(fpr_grid)
for i in range(n_classes):
    mean_tpr+=np.interp(fpr_grid,fpr[i],tpr[i])
mean_tpr/=n_classes
fpr["macro"]=fpr_grid
tpr["macro"]=mean_tpr
roc_auc["macro"]=auc(fpr["macro"],tpr["macro"])

y_true_list=list([tuple(t)for t in y_true])
classNum=dict((a,y_true_list.count(a)) for a in y_true_list)
n1=classNum[(1,0,0)]
n2=classNum[(0,1,0)]
n3=classNum[(0,0,1)]
ratio=[n1/(n1+n2+n3),n2/(n1+n2+n3),n3/(n1+n2+n3)]
avg_tpr=np.zeros_like(fpr_grid)
for i in range(n_classes):
    avg_tpr +=ratio[i]*np.interp(fpr_grid,fpr[i],tpr[i])
fpr["weighted"]=fpr_grid
tpr["weighted"]=avg_tpr
roc_auc["weighted"]=auc(fpr["weighted"],tpr["weighted"])

# 绘制ROC曲线
plt.figure(figsize=(10, 8))

# 绘制每个类别的ROC曲线
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'
                                                      ''.format(i, roc_auc[i]))

# 绘制微平均ROC曲线
plt.plot(fpr["micro"], tpr["micro"], color='gold', lw=2, linestyle=':',
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]))

# 绘制宏平均ROC曲线
plt.plot(fpr["macro"], tpr["macro"], color='deeppink', lw=2, linestyle='--',
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]))

# 绘制加权平均ROC曲线
plt.plot(fpr["weighted"], tpr["weighted"], color='cyan', lw=2, linestyle='-.',
         label='weighted-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["weighted"]))

# 绘制对角线
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Example')
plt.legend(loc="lower right")
plt.show()
