import pandas as pd
from math import log
import matplotlib.pyplot as plt
import matplotlib.image as mping
import os

image_path = 'E:/qq/'
path = 'E:/qq/watermelon2.0.txt'
data = pd.read_csv(path)

train_data = data.drop('编号', axis=1)

cc = data.iloc[:, -1]

from collections import Counter


class Node:
    def __init__(self, root=True, label=None, attri_name=None):
        self.root = root
        self.label = label
        self.attri_name = attri_name
        self.tree = {}
        self.result = {
            'label:': self.label,
            'attri_name': self.attri_name,
            'tree': self.tree,
            'root': self.root
        }

    def add_node(self, val, node):
        self.tree[val] = node

    def __repr__(self):
        return '{}'.format(self.result)


class ID3_DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    # 熵
    def calcEnt(self, y_label):
        num_samples = y_label.shape[0]
        cnt = Counter(y_label)  ## return a dictionary
        ent = -sum([(p / num_samples) * log(p / num_samples, 2) for p in cnt.values()])
        return ent

    # 条件熵

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
                print(opt_attr, '的信息增益为', infoGain)
        print(opt_attr, '的信息增益最大，选择', opt_attr)
        return opt_attr, infoGain

    def train(self, train_data, num):
        """
        input:数据集D(DataFrame格式)，特征集A，阈值eta
        output:决策树T
        """
        y_label = train_data.iloc[:, -1]
        feature_space = train_data.columns[:-1]
        features = train_data.iloc[:, :-1]


        if len(y_label.value_counts()) == 1:
            return Node(root=True, label=y_label.iloc[0])


        if len(feature_space) == 0:
            return Node(root=True, label=y_label.value_counts().sort_values(ascending=False).index[0])


        print('第', num, '次分支:')
        opt_attr_name, max_infoGain = self.OptAttri(train_data)
        num += 1


        if max_infoGain < self.epsilon:
            return Node(root=True, label=y_label.value_counts().sort_values(ascending=False).index[0])


        node_tree = Node(root=False, attri_name=opt_attr_name)
        feature_list = train_data[opt_attr_name].value_counts().index


        for f in feature_list:
            sub_train_df = train_data.loc[train_data[opt_attr_name] == f].drop([opt_attr_name], axis=1)


            sub_tree = self.train(sub_train_df, num)
            node_tree.add_node(f, sub_tree)
        return node_tree

    def fit(self, train_data, num):
        Dtree = self.train(train_data, num)
        return Dtree


import pydotplus
from pydotplus import graphviz


def tree2graph(i, g, root):
    if root.attri_name == None:
        g_node_label = f'Node:{i}\n属性:{root.attri_name}\n标签:{root.label}'
    else:
        g_node_label = f'Node:{i}\n属性:{root.attri_name}\n标签:{root.label}'
    g_node = i
    g.add_node(graphviz.Node(g_node, label=g_node_label, fontname='FangSong'))

    for val in list(root.tree):
        i, g_children = tree2graph(i + 1, g, root.tree[val])
        g.add_edge(graphviz.Edge(g_node, g_children, label=val, fontname='FangSong'))
    return i, g_node


def draw(root):
    graph = pydotplus.graphviz.Dot(graph_type='graph')
    tree2graph(0, graph, root)
    output_path = os.path.join(image_path, 'result.png')
    graph.write_png(output_path)
    image = mping.imread('E:/qq/result.png')
    plt.imshow(image)
    plt.show()


# 拟合决策树
num = 1
dt = ID3_DTree()
Dtree = dt.fit(train_data, num)

# 画树的结构
draw(Dtree)
test_data = train_data.iloc[1:12, :-1]
