import numpy as np
import pandas as pd
from math import log
from collections import Counter
from pydotplus import graphviz
# 构建数据集
data = pd.DataFrame([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 0, 1, 2, 1],
    [1, 0, 1, 2, 1],
    [2, 0, 1, 2, 1],
    [2, 0, 1, 1, 1],
    [2, 1, 0, 1, 1],
    [2, 1, 0, 2, 1],
    [2, 0, 0, 0, 0]
])
data.columns = ['年龄', '有工作', '有房子', '信用', '类别']

# 决策树节点类
class Node:
    def __init__(self, root=True, label=None, attri_name=None):
        self.root = root
        self.label = label
        self.attri_name = attri_name
        self.tree = {}
        self.result = {
            'label': self.label,
            'attri_name': self.attri_name,
            'tree': self.tree,
            'root': self.root
        }

    def add_node(self, val, node):
        self.tree[val] = node

    def __repr__(self):
        return '{}'.format(self.result)

# 定义 tree2graph 和 draw 函数
def tree2graph(i, g, root):
    g_node_label = f'Node:{i}\n属性:{root.attri_name if root.attri_name is not None else "None"}\n标签:{root.label}'
    g_node = i
    g.add_node(graphviz.Node(g_node, label=g_node_label, fontname='FangSong'))

    current_index = i
    for val in list(root.tree):
        current_index += 1
        current_index, g_children = tree2graph(current_index, g, root.tree[val])
        g.add_edge(graphviz.Edge(g_node, g_children, label=val, fontname='FangSong'))

    return current_index, g_node

def draw(root, file_path="tree_ID3.png"):
    g = graphviz.Dot()
    tree2graph(0, g, root)
    g2 = graphviz.graph_from_dot_data(g.to_string())
    g2.write_png(file_path)
    print(f"决策树已保存为 {file_path}")

# 定义 ID3 决策树类
class ID3_DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    def calcEnt(self, y_label):
        num_samples = y_label.shape[0]
        cnt = Counter(y_label)
        ent = -sum([(p / num_samples) * log(p / num_samples, 2) for p in cnt.values()])
        return ent

    def condEnt(self, attri_data, y_label):
        num_samples = y_label.shape[0]
        attri_cnt = Counter(attri_data)
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
                opt_attr = train_data.columns[i]
        return opt_attr, infoGain

    def train(self, train_data):
        y_label = train_data.iloc[:, -1]
        feature_space = train_data.columns[:-1]
        features = train_data.iloc[:, :-1]

        if len(y_label.value_counts()) == 1:
            return Node(root=True, label=y_label.iloc[0])

        if len(feature_space) == 0:
            return Node(root=True, label=y_label.value_counts().sort_values(ascending=False).index[0])

        opt_attr_name, max_infoGain = self.OptAttri(train_data)

        if max_infoGain < self.epsilon:
            return Node(root=True, label=y_label.value_counts().sort_values(ascending=False).index[0])

        node_tree = Node(root=False, attri_name=opt_attr_name)
        feature_list = train_data[opt_attr_name].value_counts().index

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

# 创建并训练决策树
id3_tree = ID3_DTree(epsilon=0.05)
Dtree = id3_tree.fit(data)
# 可视化决策树并保存为 PNG 文件
draw(Dtree)
