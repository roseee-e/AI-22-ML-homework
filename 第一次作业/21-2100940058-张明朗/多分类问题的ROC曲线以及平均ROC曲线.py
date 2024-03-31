import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 真实标签
y_true = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
# 预测得分
y_pred = np.asarray([[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4], [0.6, 0.3, 0.1],
                     [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])

n_classes = len(y_true[0])
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr_grid=np.linspace(0.0,1.0,100)
# 计算每个类别的ROC曲线和AUC值
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算微观平均的ROC曲线
micro_fpr, micro_tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(micro_fpr, micro_tpr)

#计算macro平均的ROC曲线
mean_tpr = np.zeros_like(fpr_grid) 
for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i]) 
mean_tpr /= n_classes
fpr["macro"] = fpr_grid
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"],tpr["macro"])

# 计算weighted平均的ROC曲线
y_true_list=list([tuple(t) for t in y_true])
classNum= dict((a,y_true_list.count(a)) for a in y_true_list)
n1 = classNum[(1,0,0)]
n2 = classNum[(0,1,0)]
n3 = classNum[(0,0,1)]
ratio = [n1/(n1+n2+n3),n2/(n1+n2+n3),n3/(n1+n2+n3)]
avg_tpr = np.zeros_like(fpr_grid) 
for i in range(n_classes):
    avg_tpr += ratio[i]*np.interp(fpr_grid,fpr[i], tpr[i]) 
fpr["weighted"] = fpr_grid
tpr["weighted"] = avg_tpr
roc_auc["weighted"] = auc(fpr["weighted"],tpr["weighted"])
                          

# 绘制每个类别的ROC曲线
for i in range(n_classes):
    plt.subplot(2, 3, i+1)
    plt.plot(fpr[i], tpr[i], lw=2)
    plt.plot(fpr[i], tpr[i],"ro")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Class %d ROC' % (i+1))
for i in range(n_classes):
    print('Class %d ROC AUC: %0.2f ' % (i,roc_auc[i]))

# 绘制micro平均的ROC曲线
plt.subplot(2, 3, 4)
plt.plot(micro_fpr, micro_tpr, color='green', linestyle='--', lw=2)
print('Micro-average ROC AUC: %0.2f' % roc_auc["micro"])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('micro ROC')


# 绘制macro平均的ROC曲线
plt.subplot(2, 3, 5)
plt.plot(fpr["macro"], tpr["macro"], color='deeppink', linestyle=':', lw=4, label='Macro-average ROC curve (area = %0.2f)' % roc_auc["macro"])
print('Macro-average ROC AUC: %0.2f' % roc_auc["macro"])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('macro ROC')

# 绘制weighted平均的ROC曲线
plt.subplot(2, 3, 6)
plt.plot(fpr["weighted"], tpr["weighted"], color='navy', linestyle='--', lw=2, label='Weighted-average ROC curve (area = %0.2f)' % roc_auc["weighted"])
print('Weighted-average ROC AUC: %0.2f' % roc_auc["weighted"])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('weighted ROC')
plt.show()