import pandas as pd
import numpy as np
from graphviz import Digraph

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    return np.sum([(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])

def InfoGain(data, split_attribute_name, target_name):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
    return total_entropy - Weighted_Entropy

def ID3(data, originaldata, features, target_attribute_name, parent_node_class=None, tree=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        best_feature = features[np.argmax([InfoGain(data, feature, target_attribute_name) for feature in features])]
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data, originaldata, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        return tree

def plot_tree(tree, parent_name, graph):
    for split_value, subtree in tree.items():
        if isinstance(subtree, dict):  # subtree is a dict
            node_name = f"{list(subtree.keys())[0]}?"  # Get question
            graph.node(node_name, label=node_name)
            graph.edge(parent_name, node_name, label=str(split_value))
            plot_tree(subtree, node_name, graph)
        else:
            graph.node(str(subtree) + str(np.random.random()), label=str(subtree))
            graph.edge(parent_name, str(subtree) + str(np.random.random()), label=str(split_value))

data = {
    '年龄': ['青年', '青年', '青年', '青年', '青年', '中年', '中年', '中年', '中年', '中年', '老年', '老年', '老年', '老年', '老年'],
    '有工作': ['否', '否', '是', '是', '否', '否', '否', '是', '否', '否', '否', '否', '是', '是', '否'],
    '有房子': ['否', '否', '否', '是', '否', '否', '否', '是', '是', '是', '是', '是', '否', '否', '否'],
    '信用': ['一般', '好', '好', '一般', '一般', '一般', '好', '好', '非常好', '非常好', '非常好', '好', '好', '非常好', '一般'],
    '类别': ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
}
data = pd.DataFrame(data)
features = data.columns[:-1].tolist()
target = "类别"
tree = ID3(data, data, features, target)

tree_graph = Digraph(format='png')
plot_tree(tree, 'Root', tree_graph)
tree_graph.render('decision_tree')

print("决策树已生成并存储为 PNG 文件。")