import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve

tt = [['N', 'N', 'T'], ['N', 'T', 'N'], ['T', 'N', 'N'], ['N', 'N', 'T'], ['T', 'N', 'N'],
      ['N', 'T', 'N'], ['N', 'T', 'N'], ['N', 'T', 'N'], ['N', 'N', 'T'], ['N', 'T', 'N']]
pp = [[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4],
      [0.6, 0.3, 0.1], [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]]

y_true = np.array([label == 'T' for sublist in tt for label in sublist]).reshape(len(tt), -1)
y_scores = np.array(pp)

micro_tpr = 0.0
micro_fpr = 0.0
micro_pos = 0
micro_neg = 0

for scores, labels in zip(y_scores, y_true):
      fpr, tpr, thresholds = roc_curve(labels, scores)
      micro_tpr += np.sum(tpr[:-1] * (fpr[1:] - fpr[:-1]))
      micro_fpr += np.sum(fpr[1:] - fpr[:-1])
      micro_pos += np.sum(labels)
      micro_neg += np.sum(~labels)

micro_tpr /= micro_pos
micro_fpr /= micro_neg

# 绘制ROC曲线
plt.figure(figsize=(15, 10))

# 绘制每个类别的ROC曲线
colors = ['blue', 'red', 'cyan']
labels = ['Class 1', 'Class 2', 'Class 3']
for i in range(3):
      plt.subplot(2, 2, i + 1)  
      fpr, tpr, thresholds = roc_curve(y_true[:, i], y_scores[:, i])
      plt.plot(fpr, tpr, 'r',marker='o', lw=2, alpha=0.5, label=labels[i])
      plt.plot(fpr, tpr, color=colors[i])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title(f'ROC Curve for {labels[i]}')
      plt.legend(loc="lower right")

# 绘制micro-averaging的ROC曲线
plt.subplot(2, 2, 4)  
plt.plot([0, micro_fpr, 1], [0, micro_tpr, 1], color='red',marker='o', lw=2, label='Micro-average ROC')
plt.plot( [0, micro_fpr, 1], [0, micro_tpr, 1],'g')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-average ROC Curve')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

