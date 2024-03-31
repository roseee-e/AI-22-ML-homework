import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

pp = [['T', 0.90], ['T', 0.70], ['N', 0.65], ['T', 0.60], ['N', 0.50], ['N', 0.40], ['N', 0.40], ['T', 0.40],
      ['N', 0.35], ['N', 0.20]]

y_true = []       # the real label
y_score = []      # the predicted probability
for p in pp:
      y_c = p[0]
      if y_c == 'T':
            y = 1
      else:
            y = 0

      y_hat = p[1]
      y_true.append(y)
      y_score.append(y_hat)

auc = roc_auc_score(y_true, y_score, sample_weight=None)

fpr, tpr, th = roc_curve(y_true, y_score)

plt.plot(fpr, tpr)
plt.title("ROC curve", fontsize=14)
plt.ylabel('TPR', fontsize=14)
plt.xlabel('FPR', fontsize=14)

plt.show()
