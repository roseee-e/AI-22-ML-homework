from collections import Counter
from IPython.display import display
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from math import log
from sklearn.metrics import accuracy_score
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 查找系统中支持中文的字体
font_path = 'C:/Windows/Fonts/simsun.ttc'  # 或者 'simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)

# 设置 matplotlib 使用该字体
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False


# 使用 networkx 绘制决策树
def draw(root):
    def add_edges(graph, root, parent=None, edge_label=None):
        node_label = f"{root.attri_name}\n{root.label}" if root.attri_name else str(root.label)
        graph.add_node(node_label)
        if parent:
            graph.add_edge(parent, node_label, label=edge_label)
        for val, child in root.tree.items():
            add_edges(graph, child, node_label, val)

    graph = nx.DiGraph()
    add_edges(graph, root)

    pos = nx.spring_layout(graph)
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold',
            font_color='black', font_family=font_prop.get_name())
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red',
                                 font_family=font_prop.get_name())
    plt.show()



path = 'C:/Users/HONOR/Documents/Tencent Files/704792581/FileRecv/datafortest.csv'
data = pd.read_csv(path)
train_data = data.drop(['编号'], axis=1)


# 计算熵
def calcEnt(y_label):
    num_samples = y_label.shape[0]
    cnt = Counter(y_label)
    ent = -sum([(p / num_samples) * log(p / num_samples, 2) for p in cnt.values()])
    return ent


# 计算条件熵
def condEnt(attri_data, y_label):
    num_samples = y_label.shape[0]
    attri_cnt = Counter(attri_data)
    cond_ent = 0
    for key in attri_cnt:
        attri_key_label = y_label[attri_data == key]
        cond_ent += len(attri_key_label) / num_samples * calcEnt(attri_key_label)
    return cond_ent


# 选择最优属性
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


# 定义节点类
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


# 使用networkx和matplotlib可视化决策树
def draw(root):
    def add_edges(graph, root, parent=None, edge_label=None):
        node_label = f"{root.attri_name}\n{root.label}" if root.attri_name else str(root.label)
        graph.add_node(node_label)
        if parent:
            graph.add_edge(parent, node_label, label=edge_label)
        for val, child in root.tree.items():
            add_edges(graph, child, node_label, val)

    graph = nx.DiGraph()
    add_edges(graph, root)

    pos = nx.spring_layout(graph)
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold',
            font_color='black')
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')
    plt.show()


# 定义ID3决策树类
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
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, root, test_data_Frame):
        root0 = root
        testNum = test_data_Frame.shape[0]
        pred = []
        for i in range(testNum):
            root = root0
            test_data = test_data_Frame.iloc[i:i + 1, :]
            while root.attri_name != None:
                attri_val = test_data[root.attri_name].values[0]
                if attri_val in root.tree:
                    root = root.tree[attri_val]
                else:
                    break
            y_pred = root.label
            pred.append(y_pred)
        return pred


# 创建和训练决策树
tree = ID3_DTree(epsilon=0.1)
dtree = tree.fit(train_data)

# 可视化决策树
draw(dtree)


test_data = train_data.drop(['类别'], axis=1)
predictions = tree.predict(dtree, test_data)

# 计算准确度
accuracy = accuracy_score(data['类别'], predictions)
print(f'Accuracy: {accuracy:.2f}')

