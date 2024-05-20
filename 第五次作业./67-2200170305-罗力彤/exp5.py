import pandas as pd
import numpy as np
from math import log2
from graphviz import Digraph

# 定义训练数据集
data = {
    '年龄': ['青年', '青年', '青年', '青年', '青年', '中年', '中年', '中年', '中年', '中年', '老年', '老年', '老年', '老年', '老年'],
    '有工作': ['否', '否', '是', '是', '否', '否', '否', '是', '否', '否', '否', '否', '是', '是', '否'],
    '有房子': ['否', '否', '否', '是', '否', '否', '否', '是', '是', '是', '是', '是', '否', '否', '否'],
    '信用': ['一般', '好', '好', '一般', '一般', '一般', '好', '好', '非常好', '非常好', '非常好', '好', '好', '非常好', '一般'],
    '类别': ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
}

df = pd.DataFrame(data)

# 计算数据集的信息熵
def entropy(data):
    labels = data['类别']
    label_counts = labels.value_counts()
    total = len(labels)
    entropy = sum([-count/total * log2(count/total) for count in label_counts])
    return entropy

# 计算属性的信息增益
def information_gain(data, attribute):
    total_entropy = entropy(data)
    values, counts = np.unique(data[attribute], return_counts=True)
    weighted_entropy = sum([count/len(data) * entropy(data[data[attribute] == value]) for value, count in zip(values, counts)])
    information_gain = total_entropy - weighted_entropy
    return information_gain

# 根据信息增益构建决策树
def build_decision_tree(data, parent_label=None):
    if len(data.columns) == 1:
        return parent_label
    
    gains = {attribute: information_gain(data, attribute) for attribute in data.columns if attribute != '类别'}
    max_gain_attribute = max(gains, key=gains.get)
    
    tree = {max_gain_attribute: {}}
    values = data[max_gain_attribute].unique()
    for value in values:
        subset_data = data[data[max_gain_attribute] == value].drop(columns=[max_gain_attribute])
        subtree = build_decision_tree(subset_data, parent_label=data.loc[data[max_gain_attribute].first_valid_index(), '类别'])
        tree[max_gain_attribute][value] = subtree
    
    return tree

# 构建决策树
decision_tree = build_decision_tree(df)

# 生成可视化图形
def visualize_tree(tree, dot=None):
    if dot is None:
        dot = Digraph()
    
    for attribute, subtree in tree.items():
        if isinstance(subtree, dict):
            for value, subsubtree in subtree.items():
                dot.node(f'{attribute}_{value}', label=f'{attribute}={value}')
                if isinstance(subsubtree, dict):
                    visualize_tree(subsubtree, dot)
                else:
                    dot.node(subsubtree, label=subsubtree, shape='oval')
                dot.edge(f'{attribute}_{value}', subsubtree)
    
    return dot

dot = visualize_tree(decision_tree)
dot.render('decision_tree', format='png', cleanup=True)
print('Decision tree visualization generated as decision_tree.png')