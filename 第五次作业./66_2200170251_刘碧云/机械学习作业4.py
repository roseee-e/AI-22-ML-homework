import os
import pandas as pd
import numpy as np
import graphviz
from IPython.display import Image

# 文件路径
file_path = r"D:\QQ缓存文件\Data.txt"

# 加载数据并处理异常
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    raise Exception(f"文件未找到：{file_path}")
except pd.errors.EmptyDataError:
    raise Exception("文件为空")
except Exception as e:
    raise Exception(f"读取文件时出错：{e}")

# 计算熵的函数
def calculate_entropy(data):
    labels = data.iloc[:, -1]
    label_counts = labels.value_counts()
    probabilities = label_counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# 计算整个数据集的熵
total_entropy = calculate_entropy(df)

# 计算属性熵的函数
def calculate_attribute_entropy(data, attribute):
    unique_values = data[attribute].unique()
    attribute_entropy = 0.0

    for value in unique_values:
        subset = data[data[attribute] == value]
        subset_entropy = calculate_entropy(subset)
        weight = len(subset) / len(data)
        attribute_entropy += weight * subset_entropy

    return attribute_entropy

# 计算信息增益的函数
def calculate_information_gain(data, attribute, total_entropy):
    attribute_entropy = calculate_attribute_entropy(data, attribute)
    information_gain = total_entropy - attribute_entropy
    return information_gain

# 定义决策树节点类
class DecisionTreeNode:
    def __init__(self, attribute=None, value=None, label=None, branches=None):
        self.attribute = attribute
        self.value = value
        self.label = label
        self.branches = branches if branches is not None else {}

# 构建决策树的递归函数
def id3(data, attributes):
    labels = data.iloc[:, -1]

    # 如果所有实例都有相同的标签，返回该标签
    if len(labels.unique()) == 1:
        return DecisionTreeNode(label=labels.iloc[0])

    # 如果没有属性可以使用，返回最多数的标签
    if len(attributes) == 0:
        return DecisionTreeNode(label=labels.mode()[0])

    # 选择信息增益最高的属性
    total_entropy = calculate_entropy(data)
    gains = {attribute: calculate_information_gain(data, attribute, total_entropy) for attribute in attributes}
    best_attribute = max(gains, key=gains.get)

    # 创建当前节点
    node = DecisionTreeNode(attribute=best_attribute)

    # 遍历最佳属性的每个值，递归构建子树
    unique_values = data[best_attribute].unique()
    for value in unique_values:
        subset = data[data[best_attribute] == value]
        subset_attributes = [attr for attr in attributes if attr != best_attribute]
        node.branches[value] = id3(subset, subset_attributes)

    return node

# 使用 ID3 算法构建决策树
attributes = df.columns[1:-1]  # 排除编号列和目标列
decision_tree = id3(df, attributes)

# 将决策树转换为 dot 格式的函数
def tree_to_dot(node, dot=None, parent=None, edge_label=None):
    if dot is None:
        dot = graphviz.Digraph()
        dot.node('root', '决策树')
        parent = 'root'

    if node.label is not None:
        dot.node(str(id(node)), label=f"标签: {node.label}")
    else:
        dot.node(str(id(node)), label=f"属性: {node.attribute}")

    if parent is not None:
        dot.edge(parent, str(id(node)), label=edge_label)

    for value, branch in node.branches.items():
        tree_to_dot(branch, dot, str(id(node)), f"值: {value}")

    return dot

# 创建输出目录（如果不存在）
output_directory = r"D:\pythonProject"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 将决策树转换为 dot 文件并渲染为图像
dot = tree_to_dot(decision_tree)
dot.format = 'png'
output_path = os.path.join(output_directory, "result")
dot.render(output_path)

# 输出 dot 对象的 source 以便检查
print(dot.source)

# 在Jupyter Notebook中展示图像
Image(filename=f"{output_path}.png")
