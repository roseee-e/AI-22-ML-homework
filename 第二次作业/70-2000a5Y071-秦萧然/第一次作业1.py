import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib
from sklearn.metrics import precision_recall_curve
matplotlib.use('TkAgg')  # 使用TkAgg作为后端
import matplotlib.pyplot as plt
Y_true=[1,0,0,1,0,0,1,1,0,0]
Y_score=[0.9,0.4,0.2,0.6,0.5,0.4,0.7,0.4,0.65,0.35]
length=len(Y_true)
fpr = {i: [] for i in range(length)}
tpr = {i: [] for i in range(length)}
roc_auc = {i: [] for i in range(length)}
for i in range(length):
    fpr[i], tpr[i], _ = roc_curve(Y_true,Y_score)
    roc_auc[i] = auc(fpr[i], tpr[i])
precision, recall, _ = precision_recall_curve(Y_true,Y_score)
plt.plot(fpr[i], tpr[i], label='Class %d (area = %0.2f)' % (i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Extension of ROC to multi-class')
plt.legend(loc="lower right")
plt.show()

# 计算AUC（Area Under Curve）
auc_value = auc(recall, precision)

# 绘制PR曲线
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % auc_value)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()