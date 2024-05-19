import numpy as np
import pandas as pd
from math import log
from collections import Counter
import graphviz
from IPython.display import display

# 定义数据集
data = {
    '年龄': ['青年', '青年', '青年', '青年', '青年', '中年', '中年', '中年', '中年', '中年', '老年', '老年', '老年', '老年', '老年'],
    '有工作': ['否', '否', '是', '是', '否', '否', '否', '是', '否', '否', '否', '否', '是', '是', '否'],
    '有房子': ['否', '否', '否', '是', '否', '否', '否', '是', '是', '是', '是', '是', '否', '否', '否'],
    '信用': ['一般', '好', '好', '一般', '一般', '一般', '好', '好', '非常好', '非常好', '非常好', '好', '好', '非常好', '一般'],
    '类别': ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
}

# 创建DataFrame
df = pd.DataFrame(data)

# 定义熵函数
def calcEnt(y_label):
    num_samples = y_label.shape[0]
    cnt = Counter(y_label)  # return a dictionary
    ent = -sum([(p / num_samples) * log(p / num_samples, 2) for p in cnt.values()])
    return ent

# 定义条件熵函数
def condEnt(attri_data, y_label):
    num_samples = y_label.shape[0]
    attri_cnt = Counter(attri_data)  # return a dictionary
    cond_ent = 0
    for key in attri_cnt:
        attri_key_label = y_label[attri_data == key]
        cond_ent += len(attri_key_label) / num_samples * calcEnt(attri_key_label)
    return cond_ent

# 定义函数以选择最佳属性
def OptAttri(train_data):
    infoGain = 0
    y_label = train_data.iloc[:, -1]
    attri_num = train_data.shape[1] - 1
    for i in range(attri_num):
        attri_data = train_data.iloc[:, i]
        ent = calcEnt(y_label)
        cond_ent = condEnt(attri_data, y_label)
        infoGain_tmp = ent - cond_ent
        if infoGain_tmp > infoGain:
            infoGain = infoGain_tmp
            opt_attr_name = train_data.columns[i]
    return opt_attr_name, infoGain

# 定义节点类
class Node:
    def __init__(self, root=True, label=None, attri_name=None):
        self.root = root  # Whether it's a root node
        self.label = label  # Node label
        self.attri_name = attri_name  # Node attribute name
        self.tree = {}  # Subtree of the node
        self.result = {
            'label': self.label,
            'attri_name': self.attri_name,
            'tree': self.tree,
            'root': self.root
        }

    def add_node(self, val, node):  # Add node based on attribute value
        self.tree[val] = node

    def __repr__(self):
        return '{}'.format(self.result)

# 在节点类之外定义tree2graph函数
def tree2graph(i, g, root):
    g_node_label = f'Node:{i}\n属性:{root.attri_name}\n标签:{root.label}'
    g_node = f'Node_{i}'
    g.node(g_node, label=g_node_label, fontname='FangSong')
    for val in list(root.tree):
        i, g_children = tree2graph(i + 1, g, root.tree[val])
        g.edge(g_node, g_children, label=str(val), fontname='FangSong')
    return i, g_node

def draw(root):
    g = graphviz.Digraph()
    tree2graph(0, g, root)
    display(g)

# 定义ID3决策树类
class ID3_DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    def calcEnt(self, y_label):
        num_samples = y_label.shape[0]
        cnt = Counter(y_label)  # return a dictionary
        ent = -sum([(p / num_samples) * log(p / num_samples, 2) for p in cnt.values()])
        return ent

    def condEnt(self, attri_data, y_label):
        num_samples = y_label.shape[0]
        attri_cnt = Counter(attri_data)  # return a dictionary
        cond_ent = 0
        for key in attri_cnt:
            attri_key_label = y_label[attri_data == key]
            cond_ent += len(attri_key_label) / num_samples * self.calcEnt(attri_key_label)
        return cond_ent

    def OptAttri(self, train_data):
        infoGain = 0
        y_label = train_data.iloc[:, -1]
        attri_num = train_data.shape[1] - 1
        for i in range(attri_num):
            attri_data = train_data.iloc[:, i]
            ent = self.calcEnt(y_label)
            cond_ent = self.condEnt(attri_data, y_label)
            infoGain_tmp = ent - cond_ent
            if infoGain_tmp > infoGain:
                infoGain = infoGain_tmp
                opt_attr = train_data.columns[i]  # attribute name
        return opt_attr, infoGain

    def train(self, train_data):
        y_label = train_data.iloc[:, -1]
        feature_space = train_data.columns[:-1]  # feature names

        features = train_data.iloc[:, :-1]

        if len(y_label.value_counts()) == 1:
            return Node(root=True, label=y_label.iloc[0])

        if len(feature_space) == 0:
            return Node(root=True, label=y_label.value_counts().sort_values(ascending=False).index[0])

        opt_attr_name, max_infoGain = self.OptAttri(train_data)

        if max_infoGain < self.epsilon:
            return Node(root=True, label=y_label.value_counts().sort_values(ascending=False).index[0])

        node_tree = Node(root=False, attri_name=opt_attr_name)
        feature_list = train_data[opt_attr_name].value_counts().index  # attribute values

        for f in feature_list:
            sub_train_df = train_data.loc[train_data[opt_attr_name] == f].drop([opt_attr_name], axis=1)

            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)
        return node_tree

    def fit(self, train_data):
        Dtree = self.train(train_data)
        return Dtree

    def predict(self, root, test_data_Frame):
        root0 = root
        testNum = test_data_Frame.shape[0]
        pred = []
        for i in range(testNum):
            root = root0
            test_data = test_data_Frame.iloc[i:i + 1, :]

            while root.attri_name is not None:
                attri_val = test_data[root.attri_name].values[0]
                if attri_val in root.tree:
                    root = root.tree[attri_val]
                else:
                    break
            y_pred = root.label
            pred.append(y_pred)
        return pred

# 创建ID3决策树的实例并拟合模型
tree = ID3_DTree(epsilon=0.1)
Dtree = tree.fit(df)

# 画出决策树
draw(Dtree)
