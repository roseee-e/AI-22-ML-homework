import pandas as pd
import numpy as np
from math import log2
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def entropy(s):
    p_data = s.value_counts() / len(s)
    entropy = -sum(p_data * np.log2(p_data))
    return entropy

def info_gain(data, split_name, target_name):
    total_entropy = entropy(data[target_name])
    values, counts = np.unique(data[split_name], return_counts=True)
    weighted_entropy = sum([(counts[i] / sum(counts)) * entropy(data.where(data[split_name] == values[i]).dropna()[target_name]) for i in range(len(values))])
    information_gain = total_entropy - weighted_entropy
    return information_gain

data = {
    '年龄': ['青年', '青年', '青年', '青年', '青年', '中年', '中年', '中年', '中年', '中年', '老年', '老年', '老年', '老年', '老年'],
    '有工作': ['否', '否', '是', '是', '否', '否', '否', '是', '否', '否', '否', '否', '是', '是', '否'],
    '有房子': ['否', '否', '否', '是', '否', '否', '否', '是', '是', '是', '是', '是', '否', '否', '否'],
    '信用': ['一般', '好', '好', '一般', '一般', '一般', '好', '好', '非常好', '非常好', '非常好', '好', '好', '非常好', '一般'],
    '类别': ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
}

df = pd.DataFrame(data)

columns = list(df.columns)
columns.remove('类别')
information_gains = {column: info_gain(df, column, '类别') for column in columns}
best_attribute = max(information_gains, key=information_gains.get)

X = df.drop('类别', axis=1)
y = df['类别']

label_encoders = {}
for column in X.columns:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)

plt.rcParams['font.family'] = 'SimHei'  # 设置中文字体支持
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=X.columns, class_names=['拒绝', '通过'], filled=True)
plt.show()
