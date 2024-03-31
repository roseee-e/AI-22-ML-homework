# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 定义数据
Y_true = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0],
                    [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
Y_score = np.array([[0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4],
                     [0.6, 0.3, 0.1], [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])

# 计算每个类别的ROC曲线
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(Y_true.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(Y_true[:, i], Y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算微平均、宏平均和加权平均的ROC曲线
# 微平均
fpr_micro, tpr_micro, _ = roc_curve(Y_true.ravel(), Y_score.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# 宏平均
total_fpr = np.unique(np.concatenate([fpr[i] for i in range(Y_true.shape[1])]))
m_tpr = np.zeros_like(total_fpr)
for i in range(Y_true.shape[1]):
    m_tpr += np.interp(total_fpr, fpr[i], tpr[i])
m_tpr /= Y_true.shape[1]
fpr_macro = total_fpr
tpr_macro = m_tpr
roc_auc_macro = auc(fpr_macro, tpr_macro)

# 加权平均
sample_count = np.sum(Y_true, axis=0)
weighted_sum_tpr = np.zeros_like(total_fpr)
for i in range(Y_true.shape[1]):
    weighted_sum_tpr += np.interp(total_fpr, fpr[i], tpr[i]) * sample_count[i]
weighted_avg_tpr = weighted_sum_tpr / np.sum(sample_count)
fpr_weighted_avg = total_fpr
tpr_weighted_avg = weighted_avg_tpr
roc_auc_weighted_avg = auc(fpr_weighted_avg, tpr_weighted_avg)

# 绘制每个类别的ROC曲线
plt.figure()
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
for i in range(Y_true.shape[1]):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {} (area = {:0.2f})'.format(i, roc_auc[i]))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curves')
plt.legend(loc="lower right")
plt.show()

# 绘制平均ROC曲线
plt.figure()
plt.plot(fpr_micro, tpr_micro, color='deeplink', lw=2, linestyle=':', label='Micro-average ROC curve (area = {:0.2f})'.format(roc_auc_micro))
plt.plot(fpr_macro, tpr_macro, color='navy', lw=2, linestyle=':', label='Macro-average ROC curve (area = {:0.2f})'.format(roc_auc_macro))
plt.plot(fpr_weighted_avg, tpr_weighted_avg, color='dark orange', lw=2, linestyle=':', label='Weighted-average ROC curve (area = {:0.2f})'.format(roc_auc_weighted_avg))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Average ROC curves')
plt.legend(loc="lower right")
plt.show()

