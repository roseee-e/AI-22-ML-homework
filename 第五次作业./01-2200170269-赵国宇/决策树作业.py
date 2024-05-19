import numpy as np
import pandas as pd
import math
from math import log
from sklearn.metrics import accuracy_score
import re

path =r'机器学习作业/dtreedata.csv'
data = pd.read_csv(path)
train_data = data.drop(['编号'], axis=1)  # 去掉“编号”一列

from collections import Counter#计算列表中每个元素出现的次数

def calcEnt(y_label):# 计算数据集的熵
    num_samples = y_label.shape[0]
    cnt = Counter(y_label) #返回一个字典
    ent = -sum([(p / num_samples) * log(p / num_samples, 2) for p in cnt.values()])
    return ent
def condEnt(attri_data, y_label):# 计算给定某个特征属性后的条件熵
    num_samples = y_label.shape[0]
    attri_cnt = Counter(attri_data) 
    cond_ent = 0
    for key in attri_cnt:
        attri_key_label = y_label[attri_data==key]#返回特征属性中某个特征值的标签
        cond_ent += len(attri_key_label)/num_samples * calcEnt(attri_key_label)
    return cond_ent
def OptAttri(train_data):#选择最大信息增益特征属性
    infoGain = 0  # 初始化信息增益为0
    y_label = train_data.iloc[:, -1]  # 获取标签列
    attri_num = train_data.shape[1] - 1  # 获取特征属性的数量
    for i in range (attri_num):
        attri_data = train_data.iloc[:,i]
        ent = calcEnt(y_label)#数据集的熵
        cond_ent = condEnt(attri_data, y_label)#给定特征属性后的条件熵
        infoGain_tmp = ent-cond_ent#信息增益
        if infoGain_tmp > infoGain:
            infoGain = infoGain_tmp
            opt_attr_name = train_data.columns[i]# 最优特征属性的名称
            opt_attr = i# 最优特征属性的索引
    return opt_attr, opt_attr_name, infoGain
# 定义节点类 二叉树
class Node:
    def __init__(self, root=True, label=None, attri_name=None):#初始化
        self.root = root # 是否为根节点
        self.label = label # 节点的标签
        self.attri_name = attri_name ## 节点的属性名字
        self.tree = {} # 存储子节点的字典，键是属性值，值是对应的子节点
        self.result = {
            'label:': self.label,
            'attri_name':self.attri_name,
            'tree': self.tree,
            'root': self.root
        }#结果字典存储节点信息

    def add_node(self, val, node): ## 根据属性的划分取值val，继续建立节点。
        self.tree[val] = node
    
    def __repr__(self):# 定义对象的打印表示，返回result字典的字符串格式
        return '{}'.format(self.result)

from pydotplus import graphviz
import re
from IPython.display import display, Image

# 定义函数将决策树转换为图形
def tree2graph(i, g, root):
    # 判断节点是否为叶节点（没有属性名）
    if root.attri_name == None:
        # 为叶节点创建图形节点标签
        g_node_label = f'Node:{i}\n属性:{root.attri_name}\n标签:{root.label}'
    else:
        # 为决策节点创建图形节点标签
        g_node_label = f'Node:{i}\n属性:{root.attri_name}\n标签:{root.label}'
    g_node = i
    # 在图形中添加节点
    g.add_node(graphviz.Node(g_node, label=g_node_label, fontname='FangSong'))
    
    # 遍历当前节点的子节点
    for val in list(root.tree):
        # 递归调用函数为子节点创建图形节点
        i, g_children = tree2graph(i+1, g, root.tree[val])
        # 在图形中添加边，连接当前节点和子节点
        g.add_edge(graphviz.Edge(g_node, g_children, label=val, fontname='FangSong'))
    return i, g_node

# 定义函数绘制决策树并保存为 PNG 文件
def draw(root, filename='tree.png'):
    g = graphviz.Dot()  # 创建一个新的图形
    tree2graph(0, g, root)  # 调用 tree2graph 函数开始绘制
    g2 = graphviz.graph_from_dot_data(g.to_string())  # 从 DOT 数据创建图形
    image = g2.create_png()  # 生成 PNG 图像
    # 打开文件并写入图像数据
    with open(filename, 'wb') as f:
        f.write(image)
    # 打印保存成功的消息
    print(f"决策树图已保存为 {filename}。")

# ID3决策树类
class ID3_DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon  # 信息增益的阈值
        self._tree = {}  # 存储决策树

    # 计算熵
    def calcEnt(self, y_label):
        num_samples = y_label.shape[0]  # 样本数
        cnt = Counter(y_label)  # 统计标签的频数
        # 计算熵
        ent = -sum([(p / num_samples) * log(p / num_samples, 2) for p in cnt.values()])
        return ent

    # 计算条件熵
    def condEnt(self, attri_data, y_label):
        num_samples = y_label.shape[0]  # 样本数
        attri_cnt = Counter(attri_data)  # 统计属性的频数
        cond_ent = 0
        # 计算条件熵
        for key in attri_cnt:
            attri_key_label = y_label[attri_data == key]
            cond_ent += len(attri_key_label) / num_samples * self.calcEnt(attri_key_label)
        return cond_ent

    # 选择最大信息增益的特征属性
    def OptAttri(self, train_data):
        global opt_attr
        infoGain = 0  # 初始化信息增益
        y_label = train_data.iloc[:, -1]  # 获取标签列
        attri_num = train_data.shape[1] - 1  # 特征属性数量
        # 遍历所有特征属性
        for i in range(attri_num):
            attri_data = train_data.iloc[:, i]
            ent = self.calcEnt(y_label)  # 计算熵
            cond_ent = self.condEnt(attri_data, y_label)  # 计算条件熵
            infoGain_tmp = ent - cond_ent  # 计算信息增益
            # 更新最大信息增益和最优特征属性
            if infoGain_tmp > infoGain:
                infoGain = infoGain_tmp
                opt_attr = train_data.columns[i]  # 特征属性名称
        return opt_attr, infoGain

    # 训练决策树
    def train(self, train_data):
        y_label = train_data.iloc[:, -1]  # 获取标签列
        feature_space = train_data.columns[:-1]  # 特征空间

        # 如果所有实例属于同一类，则返回单节点树
        if len(y_label.value_counts()) == 1:
            return Node(root=True, label=y_label.iloc[0])

        # 如果特征属性为空，则返回单节点树，并将样本数最多的类作为节点标签
        if len(feature_space) == 0:
            return Node(root=True, label=y_label.value_counts().sort_values(ascending=False).index[0])

        # 选择最大信息增益的特征属性
        opt_attr_name, max_infoGain = self.OptAttri(train_data)

        # 如果信息增益小于阈值，则返回单节点树，并将样本数最多的类作为节点标签
        if max_infoGain < self.epsilon:
            return Node(root=True, label=y_label.value_counts().sort_values(ascending=False).index[0])

        # 创建决策节点
        node_tree = Node(root=False, attri_name=opt_attr_name)
        feature_list = train_data[opt_attr_name].value_counts().index
        # 遍历特征属性的所有可能值
        for f in feature_list:
            # 创建子训练集
            sub_train_df = train_data.loc[train_data[opt_attr_name] == f].drop([opt_attr_name], axis=1)
            # 递归构建子树
            sub_tree = self.train(sub_train_df)
            # 将子树添加到决策节点
            node_tree.add_node(f, sub_tree)
        return node_tree

    # 拟合决策树
    def fit(self, train_data):
        Dtree = self.train(train_data)
        return Dtree

    # 预测函数
    def predict(self, root, test_data_Frame):
        root0 = root  # 保存根节点
        testNum = test_data_Frame.shape[0]  # 测试数据数量
        pred = []  # 预测结果列表
        # 遍历测试数据
        for i in range(testNum):
            root = root0  # 重置为根节点
            test_data = test_data_Frame.iloc[i:i + 1, :]  # 获取单个测试实例

            # 遍历决策树直到叶节点
            while root.attri_name != None:
                attri_val = test_data[root.attri_name].values[0]  # 获取属性值
                # 如果属性值在子树中，则继续遍历
                if attri_val in root.tree:
                    root = root.tree[attri_val]
                else:
                    break
            y_pred = root.label  # 获取预测标签
            pred.append(y_pred)  # 添加到预测结果列表
        return pred

# 拟合决策树
dt = ID3_DTree()
Dtree = dt.fit(train_data)

# 画树的结构
draw(Dtree, 'my_tree.png')
test_data = train_data.iloc[1:12,:-1]

# # 预测结果
y_pred = dt.predict(Dtree,test_data)
print (y_pred)


