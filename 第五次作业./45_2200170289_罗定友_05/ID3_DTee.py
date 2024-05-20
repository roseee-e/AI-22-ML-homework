import pandas as pd
from collections import Counter
import math
from math import log
from pydotplus import graphviz
from IPython.display import display, Image

# 加载数据并检查列名
path = r"D:\作业\机器学习\Homework_02\DTree_data.txt"
columns = ['编号', '年龄', '有工作', '有房子', '信用', '类别']
data = pd.read_csv(path, sep=',', names=columns, header=0)

# 打印数据和列名，确保正确
print(data.head())
print(data.columns)

# 去除列名空白字符
data.columns = data.columns.str.strip()

# 删除 '编号' 列
train_data = data.drop(['编号'], axis=1)
print(train_data.head())

# 定义计算熵、条件熵和最佳属性选择函数
def calcEnt(y_label):
    num_samples = y_label.shape[0]
    cnt = Counter(y_label)
    ent = -sum([p / num_samples * math.log(p / num_samples) / math.log(2) for p in cnt.values()])
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

# 定义决策树节点类和树的可视化函数
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
    g.write_png('decision_tree.png')
    print(f"Decision tree graph saved to decision_tree.png")

# 定义 ID3 决策树类并训练模型
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

# 计算信息增益函数
def calcIV(attri_data):
    num_samples = attri_data.shape[0]
    attri_cnt = Counter(attri_data)
    IV = 0
    for key in attri_cnt:
        IV += -attri_cnt[key] / num_samples * np.log2(attri_cnt[key] / num_samples)
    return IV

# 计算并打印每个属性的信息增益
def print_info_gains(train_data):
    y_label = train_data.iloc[:, -1]
    attributes = train_data.columns[:-1]
    info_gains = {}

    for attri_name in attributes:
        attri_data = train_data[attri_name]
        ent = calcEnt(y_label)
        cond_ent = condEnt(attri_data, y_label)
        info_gain = ent - cond_ent
        info_gains[attri_name] = info_gain
        print(f"属性 {attri_name} 的信息增益为: {info_gain}")

    return info_gains

# 确保 calcEnt 和 condEnt 函数在全局作用域中
def calcEnt(y_label):
    num_samples = y_label.shape[0]
    cnt = Counter(y_label)
    ent = -sum([p / num_samples * math.log(p / num_samples) / math.log(2) for p in cnt.values()])
    return ent

def condEnt(attri_data, y_label):
    num_samples = y_label.shape[0]
    attri_cnt = Counter(attri_data)
    cond_ent = 0
    for key in attri_cnt:
        attri_key_label = y_label[attri_data == key]
        cond_ent += len(attri_key_label) / num_samples * calcEnt(attri_key_label)
    return cond_ent

# 加载并处理数据
data = pd.read_csv(path, sep=',', names=columns, header=0)
data.columns = data.columns.str.strip()
train_data = data.drop(['编号'], axis=1)

# 拟合决策树
dt = ID3_DTree()
Dtree = dt.fit(train_data)

# 画树的结构并显示
draw(Dtree)

# 显示生成的决策树图像
display(Image(filename='decision_tree.png'))

# 计算并打印信息增益
info_gains = print_info_gains(train_data)
print(info_gains)

