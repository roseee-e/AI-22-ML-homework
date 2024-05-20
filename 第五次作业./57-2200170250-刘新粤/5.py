import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

# 数据集
data = {
    '年龄': ['青年', '青年', '青年', '青年', '青年', '中年', '中年', '中年', '中年', '中年', '老年', '老年', '老年', '老年', '老年'],
    '有工作': ['否', '否', '是', '是', '否', '否', '否', '是', '否', '否', '否', '否', '是', '是', '否'],
    '有房子': ['否', '否', '否', '是', '否', '否', '否', '是', '是', '是', '是', '是', '否', '否', '否'],
    '信用': ['一般', '好', '好', '一般', '一般', '一般', '好', '好', '非常好', '非常好', '非常好', '好', '好', '非常好', '一般'],
    '类别': ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
}

# 转换为DataFrame
df = pd.DataFrame(data)

# 将分类变量转换为数值型（对于scikit-learn）
df = pd.get_dummies(df, columns=['年龄', '有工作', '有房子', '信用'])

# 划分特征和标签
X = df.drop('类别', axis=1)
y = df['类别'].map({'否': 0, '是': 1})

# 使用DecisionTreeClassifier训练模型
clf = DecisionTreeClassifier(criterion='entropy')  # 使用熵作为划分标准
clf.fit(X, y)

# 导出决策树为文本格式
tree_str = export_text(clf, feature_names=X.columns, class_names=['否', '是'], show_weights=True)

# 打印决策树结构
print(tree_str)