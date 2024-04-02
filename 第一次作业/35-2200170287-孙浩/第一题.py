import numpy as np  
from sklearn.metrics import precision_recall_curve, roc_curve, auc  
import matplotlib.pyplot as plt  
  
# 给定的数据  
y_true = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])  # 真实标签  
y_score = np.array([0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35])  # 预测分数  
  
# 计算PR曲线的指标  
precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)  
auc_pr = auc(recalls, precisions)  
  
# 计算ROC曲线的指标  
fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)  
auc_roc = auc(fpr, tpr)  
  
# 绘制PR曲线  
plt.figure(figsize=(10, 5))  
plt.plot(recalls, precisions, color='b', label=f'PR Curve (AUC = {auc_pr:.2f})')  
plt.xlabel('Recall')  
plt.ylabel('Precision')  
plt.ylim([0.0, 1.05])  
plt.xlim([0.0, 1.0])  
plt.title('Precision-Recall Curve')  
plt.legend(loc="lower left")  
plt.grid(True)  
  
# 绘制ROC曲线（在同一个图上）  
plt.figure(figsize=(10, 5))  
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_roc:.2f})')  
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.ylim([0.0, 1.05])  
plt.xlim([0.0, 1.0])  
plt.title('Receiver Operating Characteristic Curve')  
plt.legend(loc="lower right")  
plt.grid(True)  
  
# 显示图形  
plt.show()
