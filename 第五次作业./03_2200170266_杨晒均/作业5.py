
import numpy as np
import pandas as pd
import math
from math import log
from sklearn.metrics import accuracy_score

path = 'C:/Users/杨晒均/Desktop/datetext.txt'
data = pd.read_csv(path)
train_data data.drop(['编号'], axis=1)

# 计算数据集的熵
from collections import Counter

def calc_entropy(y_label):
    num_samples = y_label.shape[0]
    cnt = Counter(y_label)
    ent = -sum([(p / num_samples) * log(p / num_samples, 2) for p in cnt.values()])
    return ent

def calc_conditional_entropy(attri_data, y_label):
    num_samples = y_label.shape[0]
    attr_cnt = Counter(attri_data)
    cond_entropy = 0
    for key in attri:
        attri_key_label = y_label[attri_data==key]
        cond_entropy len(attri_key_label) / num_samples * calc_entropy(attri_key_label)
    return cond_entropy

# 定义节点类，构建二叉树
class Node:
    def __init__(self, root=True, label=None, attri_name=None):
        self.root = root 
        self.label = label
        self.attri_name =i_name
        self.tree = {}
        self.result = {
            'label:': self.label,
            'attri_name': self.attri_name,
            'tree': self.tree,
           root': self.root
        }

    def add_node(self, val, node):  
        self.tree[val] = node

    def __repr__(self):
        return '{}'.format(self.resultclass ID3_DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.tree = {}

    def determine_opt_attr(self, train):
        info_gain = 0
        y_label = train_data.iloc[:, -1]
        attri_num = train_data.shape[1] 1
        info_gains = {}
        for i in range(attri_num            attri_data = train_data.iloc[:, i]
            ent = calc_entropy(y_label)
            cond_ent = calc_conditional_entropy(attr_data, y_label)
            info_gain_tmp = ent - cond_ent
            info_gains[train_data.columns[i]] = info_gain
            if info_gain_tmp > info_gain:
                info_gain = info_gain_tmp
                opt_attr = train_data.columns[i]
        print("各的信息增益：", info_gains)
        return opt_attr, info_gain

    def build_tree(self, train_data):
        y_label = train_data.iloc[:, -1]
        feature_space = train_data.columns[:-1]

        if len(y_label.value_counts()) == 1:
            return Node(root=True, label=y_label.iloc[0])

        if len(feature_space) == 0:
            return Node(root=True, label=y_label.value_counts().sort_values(ascending=False).index[0])

        opt_attr_name, max_info_gain = self.determine_opt_attr(train_data)

        if max_info_gain < self.epsilon:
            return Node(root=True, label=y_label.value_counts().sort_values(ascending=False).index[0])

        node_tree = Node(root=False, attri_name=opt_attr_name)
        feature_list = train_data[opt_attr_name].value_counts().index

        for f feature_list:
            sub_train_df = train_data.loc[train_data[opt_attr_name] == f].drop([opt_attr_name], axis=1            sub_tree = self.build_tree(sub_train_df)
            node_tree.add_node(f, sub_tree)
        return node_tree

    def fit(self, train_data):
        self.tree = self.build_tree(train_data)

    def predict(self, test_data):
        pred = []
        for _, row in test_data.iterrows():
            node = self.tree
            while node.attri_name is not None:
                attri_val = row[node.attri_name]
                if attri_val in node.tree:
                    = node.tree[attri_val]
                else:
                    break
            pred.append(node.label)
        pred

# 拟合决策树
dt = ID3_DTree()
dt.fit(train_data)

# 预测结果
test_data = train_data.iloc[1:12, :-1]
_pred = dt.predict(test_data)
print(y_pred)

