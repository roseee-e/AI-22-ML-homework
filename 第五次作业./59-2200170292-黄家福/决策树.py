import pandas as pd
import numpy as np
from collections import Counter
import math
import pprint
from graphviz import Digraph
from IPython.display import display

# 数据集
data = [['青年', '否', '否', '一般', '否'],
        ['青年', '否', '否', '好', '否'],
        ['青年', '是', '否', '好', '是'],
        ['青年', '是', '是', '一般', '是'],
        ['青年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '好', '否'],
        ['中年', '是', '是', '好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '好', '是'],
        ['老年', '是', '否', '好', '是'],
        ['老年', '是', '否', '非常好', '是'],
        ['老年', '否', '否', '一般', '否']]

# 创建DataFrame
df = pd.DataFrame(data, columns=['年龄', '有工作', '有房子', '信用', '类别'])

# 计算熵的函数
def entropy(data):
    label_counts = Counter(data)
    total_count = len(data)
    return -sum((count / total_count) * math.log2(count / total_count) for count in label_counts.values())

# 计算信息增益
def info_gain(df, attribute, target):
    total_entropy = entropy(df[target])
    values = df[attribute].unique()
    weighted_entropy = sum((len(subset) / len(df)) * entropy(subset[target]) for value in values for subset in [df[df[attribute] == value]])
    return total_entropy - weighted_entropy

# 选择最优属性
def best_attribute(df, attributes, target):
    return max(attributes, key=lambda attribute: info_gain(df, attribute, target))

# ID3算法生成决策树
def id3(df, target, attributes):
    # 如果所有实例都属于同一类
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]

    # 如果属性集为空，返回出现最多的类别
    if not attributes:
        return df[target].mode()[0]

    # 选择最优属性
    best_attr = best_attribute(df, attributes, target)
    tree = {best_attr: {}}

    # 对每个属性值，生成子树
    for value in df[best_attr].unique():
        sub_df = df[df[best_attr] == value]
        sub_attributes = [attr for attr in attributes if attr != best_attr]
        subtree = id3(sub_df, target, sub_attributes)
        tree[best_attr][value] = subtree

    return tree

# 可视化决策树
def visualize_tree(tree, parent_name='', graph=None):
    if graph is None:
        graph = Digraph(format='png')
        graph.attr('node', shape='ellipse')

    if isinstance(tree, dict):
        for node, branches in tree.items():
            if isinstance(branches, dict):
                for branch, subtree in branches.items():
                    child_name = f'{node}={branch}'
                    graph.node(child_name)
                    if parent_name:
                        graph.edge(parent_name, child_name)
                    visualize_tree(subtree, child_name, graph)
            else:
                leaf_name = f'{branches}'
                graph.node(leaf_name, shape='box')
                if parent_name:
                    graph.edge(parent_name, leaf_name)
    else:
        leaf_name = f'{tree}'
        graph.node(leaf_name, shape='box')
        if parent_name:
            graph.edge(parent_name, leaf_name)

    return graph

# 生成决策树
attributes = ['年龄', '有工作', '有房子', '信用']
target = '类别'
decision_tree = id3(df, target, attributes)

# 可视化决策树并显示
graph = visualize_tree(decision_tree)
display(graph)