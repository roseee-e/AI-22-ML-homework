import numpy as np

# 训练数据集
train_data = [
    {'A': 1, 'B': 0, 'C': 0, 'y': 1},
    {'A': 0, 'B': 1, 'C': 0, 'y': 0},
    {'A': 0, 'B': 1, 'C': 0, 'y': 1},
    {'A': 1, 'B': 0, 'C': 0, 'y': 1},
    {'A': 0, 'B': 1, 'C': 1, 'y': 0},
    {'A': 1, 'B': 0, 'C': 1, 'y': 0},
    {'A': 0, 'B': 0, 'C': 0, 'y': 1},
    {'A': 0, 'B': 1, 'C': 1, 'y': 0},
    {'A': 1, 'B': 0, 'C': 0, 'y': 1},
    {'A': 1, 'B': 1, 'C': 0, 'y': 1}
]

# 计算每个类别的频率
class_counts ={}
for sample in train_data:
    y = sample['y']
    if y not in class_counts:
        class_counts[y] = 1
    else:
        class_counts[y] += 1

# 计算每个特征在每个类别中的条件概率
conditional_probs = {}
for feature in ['A', 'B', 'C']:
    conditional_probs[feature] = {}
    for y in class_counts:
        conditional_probs[feature][y] = {}
        for value in [0, 1]:
            count = sum(1 for sample in train_data if sample[feature] == value and sample['y'] == y)
            total = class_counts[y]
            conditional_probs[feature][y][value] = (count + 1) / (total + 2)  # 使用拉普拉斯平滑

# 新样本
new_sample = {'A': 0, 'B': 0, 'C': 1}

# 使用朴素贝叶斯定理计算新样本属于每个类别的概率
probabilities = {}
for y in class_counts:
    probability = class_counts[y]
    for feature in ['A', 'B', 'C']:
        probability *= conditional_probs[feature][y][new_sample[feature]]
    probabilities[y] = probability

# 选择概率最高的类别作为预测结果
predicted_class = max(probabilities, key=probabilities.get)

print(f"预测结果为: {predicted_class}类")
