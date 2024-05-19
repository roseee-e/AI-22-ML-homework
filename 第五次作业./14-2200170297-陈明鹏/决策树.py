import pandas as pd
import numpy as np
import math
import pprint
from graphviz import Digraph

# 定义数据集
data = {
    '年龄': ['青年', '青年', '青年', '青年', '青年', '中年', '中年', '中年', '中年', '中年', '老年', '老年', '老年', '老年', '老年'],
    '有工作': ['否', '否', '是', '是', '否', '否', '否', '是', '否', '否', '否', '否', '是', '是', '否'],
    '有房子': ['否', '否', '否', '是', '否', '否', '否', '是', '是', '是', '是', '是', '否', '否', '否'],
    '信用': ['一般', '好', '好', '一般', '一般', '一般', '好', '好', '非常好', '非常好', '非常好', '好', '好', '非常好', '一般'],
    '类别': ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
}

# 转换为DataFrame
df = pd.DataFrame(data)

# 计算熵
def entropy(labels):
    value_counts = labels.value_counts()
    probabilities = value_counts / len(labels)
    return -sum(probabilities * np.log2(probabilities))

# 计算信息增益
def information_gain(data, split_attribute_name, target_name="类别"):
    total_entropy = entropy(data[target_name])
    values, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = sum((counts[i] / sum(counts)) * entropy(data.where(data[split_attribute_name] == values[i]).dropna()[target_name]) for i in range(len(values)))
    return total_entropy - weighted_entropy

# ID3算法
def id3(data, original_data, features, target_attribute_name="类别", parent_node_class=None):
    # 若目标属性只有一个类别，返回该类别
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    # 若数据集为空，返回父节点中出现次数最多的类别
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    
    # 若特征为空，返回父节点中出现次数最多的类别
    elif len(features) == 0:
        return parent_node_class
    
    # 若以上条件都不满足，构建树
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        # 选择信息增益最大的特征
        item_values = [information_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        # 创建子树
        tree = {best_feature: {}}
        
        # 删除已选特征
        features = [i for i in features if i != best_feature]
        
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = id3(sub_data, data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        
        return tree

# 构建决策树
features = df.columns[:-1]
target_attribute = '类别'
tree = id3(df, df, features, target_attribute)

# 打印决策树
pprint.pprint(tree)

# 可视化决策树
def visualize_tree(tree, parent_name='', graph=None):
    if graph is None:
        graph = Digraph()
        graph.attr(size='10,10')
        graph.node(name='root', label=parent_name)

    for node in tree.keys():
        if isinstance(tree[node], dict):
            for edge in tree[node].keys():
                child_name = node + '_' + edge
                graph.node(name=child_name, label=str(tree[node][edge]) if not isinstance(tree[node][edge], dict) else edge)
                graph.edge(parent_name, child_name, label=edge)
                if isinstance(tree[node][edge], dict):
                    visualize_tree(tree[node][edge], child_name, graph)
    return graph

# 可视化
graph = visualize_tree(tree, parent_name='root')
graph.view()
