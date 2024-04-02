import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

# 数据准备
Y_true = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])  # 真实标签
Y_score = np.array([0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35])  # 预测得分

# 计算精确率和召回率
precision, recall, thresholds = precision_recall_curve(Y_true, Y_score)

# 计算ROC曲线的TPR和FPR
fpr, tpr, thresholds = roc_curve(Y_true, Y_score)

# 计算AUC值
roc_auc = auc(fpr, tpr)

# 绘制PR曲线
plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # 随机分类器的对角线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 打印AUC值
print("AUC =", roc_auc)