import numpy as np
import pandas as pd
import math
from math import log
from sklearn.metrics import accuracy_score
import re
path = r"C:\Users\cycy20\Desktop\����.txt"
data = pd.read_csv(path, sep=r'\s+')
data.head()
train_data = data.drop(['���'], axis=1)
train_data.head()
from collections import Counte
def calcEnt(y_label):
    num_samples = y_label.shape[0]
    cnt = Counter(y_label)
    ent = -sum([p / num_samples * math.log(p / num_samples) / math.log(2) for p in cnt.values()])  # ʹ�� math.log ����ѧ���׹�ʽ
    return ent


def condEnt(attri_data, y_label):
    num_samples = y_label.shape[0]
    attri_cnt = Counter(attri_data)
    cond_ent = 0
    for key in attri_cnt:
        attri_key_label = y_label[attri_data == key]
        cond_ent += len(attri_key_label) / num_samples * calcEnt(attri_key_label)
    return cond_ent


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
            opt_attr = i
    return opt_attr, opt_attr_name, infoGain


class Node:
    def __init__(self, root=True, label=None, attri_name=None):
        self.root = root  ## �Ƿ�Ϊ���ڵ�
        self.label = label  ## �ڵ�ı�ǩ
        self.attri_name = attri_name  ## �ڵ����������
        self.tree = {}  ## ĳ���ڵ������
        self.result = {
            'label:': self.label,
            'attri_name': self.attri_name,
            'tree': self.tree,
            'root': self.root
        }

    def add_node(self, val, node):  ## �������ԵĻ���ȡֵval�����������ڵ㡣
        self.tree[val] = node

    def __repr__(self):
        return '{}'.format(self.result)


from pydotplus import graphviz
import re
from IPython.display import display, Image


def tree2graph(g, node, prefix=''):
    label = f'{node.attri_name}\n{node.label}'
    g_node = graphviz.Node(str(id(node)), label=label)
    g.add_node(g_node)

    for val, child_node in node.tree.items():
        child_label = f'{child_node.attri_name}\n{child_node.label}'
        g_child_node = graphviz.Node(str(id(child_node)), label=child_label)
        g.add_node(g_child_node)
        g.add_edge(graphviz.Edge(str(id(node)), str(id(child_node)), label=val))
        tree2graph(g, child_node, prefix)


def draw(root):
    if root is None:
        return
    g = graphviz.Dot()
    tree2graph(g, root)
    g.write_png('decision_tree.png')  # ����ͼ���ļ�ϵͳ
    print(f"Decision tree graph saved to decision_tree.png")


class ID3_DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    # ��
    def calcEnt(self, y_label):
        num_samples = y_label.shape[0]
        cnt = Counter(y_label)  ## return a dictionary
        ent = -sum([(p / num_samples) * log(p / num_samples, 2) for p in cnt.values()])
        return ent

    # ������

    def condEnt(self, attri_data, y_label):
        num_samples = y_label.shape[0]
        attri_cnt = Counter(attri_data)  ## return a dictionary
        cond_ent = 0
        for key in attri_cnt:
            attri_key_label = y_label[attri_data == key]
            cond_ent += len(attri_key_label) / num_samples * self.calcEnt(attri_key_label)
        return cond_ent

    ## ��������ѡ��

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
                opt_attr = train_data.columns[i]  ## attri name
        return opt_attr, infoGain

    def train(self, train_data):
        """
        input:���ݼ�D(DataFrame��ʽ)��������A����ֵeta
        output:������T
        """
        y_label = train_data.iloc[:, -1]
        feature_space = train_data.columns[:-1]  ## feature names

        features = train_data.iloc[:, :-1]

        # 1,��D��ʵ������ͬһ��Ck����y_label �е�������Ϊ1����TΪ���ڵ�����������Ck��Ϊ�������ǣ�����T
        if len(y_label.value_counts()) == 1:
            return Node(root=True, label=y_label.iloc[0])

        # 2, ����������AΪ�գ���TΪ���ڵ�������D��������������Ck��Ϊ�ýڵ�����ǣ�����T
        if len(feature_space) == 0:
            return Node(root=True, label=y_label.value_counts().sort_values(ascending=False).index[0])

        # 3,�����������������Ҫ������Ϣ����, ѡ����Ϣ��������������Ϊ��ѡ���ڵ�����Ag
        opt_attr_name, max_infoGain = self.OptAttri(train_data)

        # 4,��Ag����Ϣ����С����ֵeta,����TΪ���ڵ���������D����������������Ck��Ϊ�ýڵ�����ǣ�����T
        if max_infoGain < self.epsilon:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 5,��Ag��infogain������ֵeta������Ҫ����Ag�Ӽ�
        node_tree = Node(root=False, attri_name=opt_attr_name)
        feature_list = train_data[opt_attr_name].value_counts().index  ## ��ѡ���Ե������Ե�����

        ## ��Ҫȷ��ÿ������������������Ӽ������Ҹ����������Զ������Ӽ������������Ļ��� ��ȥ�������ԣ�
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[opt_attr_name] == f].drop([opt_attr_name], axis=1)

            # 6, �ݹ����������������Ӽ������������Ļ���
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)
        return node_tree

    def fit(self, train_data):
        Dtree = self.train(train_data)
        return Dtree

    def predict(self, root, test_data_Frame):  ## test_data_Frame ������������ӣ�û�б��������ԣ�, root �ǽ����õ���
        root0 = root
        testNum = test_data_Frame.shape[0]
        pred = []
        for i in range(testNum):
            root = root0
            test_data = test_data_Frame.iloc[i:i + 1, :]

            while root.attri_name != None:  ## ˵�����ڵ���Ի���
                attri_val = test_data[root.attri_name].values[0]  # ���ڵ������ȡֵ
                if attri_val in root.tree:  # �������������򽫴�ʱ�����Խڵ㵱�ɸ��ڵ㣬ֱ�����ܻ���ʱ����ΪҶ�ڵ�
                    root = root.tree[attri_val]
                else:
                    break
            y_pred = root.label
            pred.append(y_pred)
        return pred


def calcIV(attri_data):
    num_samples = attri_data.shape[0]
    attri_cnt = Counter(attri_data)  ## return a dictionary
    IV = 0
    for key in attri_cnt:
        IV += -attri_cnt[key] / num_samples * np.log2(attri_cnt[key] / num_samples)
    return IV

# ���㲢��ӡÿ�����Ե���Ϣ����
def print_info_gains(train_data):
    y_label = train_data.iloc[:, -1]
    attributes = train_data.columns[:-1]  # �����������ԣ���������ǩ
    info_gains = {}

    for attri_name in attributes:
        attri_data = train_data[attri_name]
        ent = calcEnt(y_label)
        cond_ent = condEnt(attri_data, y_label)
        info_gain = ent - cond_ent
        info_gains[attri_name] = info_gain
        print(f"���� {attri_name} ����Ϣ����Ϊ: {info_gain}")

    return info_gains


attri_data = train_data.iloc[:, 2]
calcIV(attri_data)
# ��Ͼ�����
# ���������������ӻ�
dt = ID3_DTree()
Dtree = dt.fit(train_data)

# �����Ľṹ
draw(Dtree)
info_gains = print_info_gains(train_data)
print(info_gains)
