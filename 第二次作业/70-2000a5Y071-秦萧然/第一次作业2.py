import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg作为后端
import matplotlib.pyplot as plt

# 真实标签和预测概率
y_true = np.asarray([[0,0,1],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0]])
y_pred = np.asarray([[0.1,0.2,0.7],[0.1,0.6,0.3],[0.5,0.2,0.3],[0.1,0.1,0.8],[0.4,0.2,0.4],[0.6,0.3,0.1],
[0.4,0.2,0.4],[0.4,0.1,0.5],[0.1,0.1,0.8],[0.1,0.8,0.1]])
# 确保y_pred中的概率和接近1
y_pred_norm = y_pred / y_pred.sum(axis=1, keepdims=True)

# 计算每个类别的ROC曲线和AUC值
n_classes = y_true.shape[1]
fpr = {i: [] for i in range(n_classes)}
tpr = {i: [] for i in range(n_classes)}
roc_auc = {i: 0 for i in range(n_classes)}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_norm[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算micro-average ROC曲线和AUC值
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred_norm.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 计算macro-average ROC曲线和AUC值
tprs = np.zeros((n_classes, len(fpr["micro"])))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

all_tprs = []

fpr_grid=np.linspace(0.0,1.0,100)
mean_tpr=np.zeros_like(fpr_grid)
for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid,fpr[i],tpr[i])
mean_tpr /= n_classes
fpr["macro"] = mean_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
y_true_list = list([tuple(t) for t in y_true])
classNum=dict((a,y_true_list.count(a)) for a in y_true_list)
n1=classNum[(1,0,0)]
n2 = classNum[(0,1,0)]
n3=classNum[(0,0,1)]
ratio = [n1/(n1+n2+n3),n2/(n1+n2+n3),n3/(n1+n2+n3)]
avg_tpr=ratio[i]*np.interp(fpr_grid,fpr[i],tpr[i])
for i in range(n_classes):
    avg_tpr += ratio[i]*np.interp(fpr_grid,fpr[i],tpr[i])
fpr["weighted"]=fpr_grid
tpr["weighted"]=avg_tpr
roc_auc["weighted"]=auc(fpr["weighted"],tpr["weighted"])
# 输出结果
print("Micro-average ROC AUC: ", roc_auc["micro"])
print("Macro-average ROC AUC: ", roc_auc["macro"])
print("Weighted-average ROC AUC: ", roc_auc["weighted"])

# 如果您想要绘制ROC曲线，可以使用matplotlib库进行绘制
import matplotlib.pyplot as plt
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='Class %d (area = %0.2f)' % (i, roc_auc[i]))

# 绘制micro-average ROC曲线
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = %0.2f)' % roc_auc["micro"])

# 绘制macro-average ROC曲线
plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = %0.2f)' % roc_auc["macro"])

# 绘制weighted-average ROC曲线
plt.plot(fpr["weighted"],tpr["weighted"], label='weighted-average ROC curve (area = %0.2f)' % roc_auc["weighted"])

# 绘制对角线
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Extension of ROC to multi-class')
plt.legend(loc="lower right")
plt.show()
