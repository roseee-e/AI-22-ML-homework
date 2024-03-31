mport numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score

# 准备数据
Y_true = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])
Y_score = np.array([0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35])

# 计算PR曲线
precision, recall, _ = precision_recall_curve(Y_true, Y_score)
plt.figure(1, figsize=(6, 6))
plt.plot(recall, precision, color='b', alpha=0.9, lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# 计算ROC曲线
fpr, tpr, _ = roc_curve(Y_true, Y_score)
plt.figure(2, figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')

# 计算AUC值
auc = roc_auc_score(Y_true, Y_score)
print("AUC值为:", auc)

# 显示图像
plt.show()