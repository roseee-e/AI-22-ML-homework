# 作者:liuqing
# 讲师:james
# 开发日期:2024/3/31
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 定义真实标签和预测标签
y_true = ['ant', 'bird', 'ant', 'cat', 'cat', 'cat']
y_pred = ['ant', 'cat', 'ant', 'cat', 'cat', 'ant']

label_map = {'ant': 0, 'bird': 1, 'cat': 2}
y_true = [label_map[label] for label in y_true]
y_pred = [label_map[label] for label in y_pred]

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# 计算精确度、召回率和F1得分
precision_micro = precision_score(y_true, y_pred, average='micro')
precision_macro = precision_score(y_true, y_pred, average='macro')
precision_weighted = precision_score(y_true, y_pred, average='weighted')

recall_micro = recall_score(y_true, y_pred, average='micro')
recall_macro = recall_score(y_true, y_pred, average='macro')
recall_weighted = recall_score(y_true, y_pred, average='weighted')

f1_micro = f1_score(y_true, y_pred, average='micro')
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print("Micro Precision:", precision_micro)
print("Macro Precision:", precision_macro)
print("Weighted Precision:", precision_weighted)
print("Micro Recall:", recall_micro)
print("Macro Recall:", recall_macro)
print("Weighted Recall:", recall_weighted)
print("Micro F1 Score:", f1_micro)
print("Macro F1 Score:", f1_macro)
print("Weighted F1 Score:", f1_weighted)

