import 练习1_二分类_不调库 as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

y_true = [1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
y_score = [0.90, 0.40, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35]

# 一个窗口中划分多个绘图区域  plt.subplots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))



# 画PR曲线
precision, recall, thresholds = precision_recall_curve(y_true, y_score)
# axs[0].set_figure("P-R Curve")
axs[0].plot(recall, precision)
axs[0].plot(recall, precision, 'ro')
axs[0].set_title('PR Curve', fontsize=14)
axs[0].set_xlabel('Recall', fontsize=14)
axs[0].set_ylabel('Precision', fontsize=14)

fpr, tpr, th = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score, sample_weight=None)
print(auc)

axs[1].plot(fpr, tpr)
axs[1].plot(fpr, tpr, 'ro')  # 画出点，并标红
axs[1].set_title("ROC curve", fontsize=14)
axs[1].set_ylabel("TPR", fontsize=14)
axs[1].set_xlabel("FPR", fontsize=14)
plt.show()
