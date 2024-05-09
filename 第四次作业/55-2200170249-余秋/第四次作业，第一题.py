# 导入collections模块中的defaultdict，用于创建字典，其中值默认为另一个字典
from collections import defaultdict

# 假设的训练数据集
train_data = [
    {'A': 1, 'B': 0, 'C': 0, 'Y': 1},
    {'A': 0, 'B': 1, 'C': 0, 'Y': 0},
    {'A': 0, 'B': 1, 'C': 0, 'Y': 1},
    {'A': 1, 'B': 0, 'C': 0, 'Y': 1},
    {'A': 0, 'B': 1, 'C': 1, 'Y': 0},
    {'A': 1, 'B': 0, 'C': 1, 'Y': 0},
    {'A': 0, 'B': 0, 'C': 0, 'Y': 1},
    {'A': 0, 'B': 1, 'C': 1, 'Y': 0},
    {'A': 1, 'B': 0, 'C': 0, 'Y': 1},
    {'A': 0, 'B': 1, 'C': 0, 'Y': 0},
]
# 类别标签集合，从数据集中提取唯一的类别标签
classes = set(data['Y'] for data in train_data)

# 计算先验概率（即每个类别的概率）
# num_samples是数据集中的样本总数
num_samples = len(train_data)
# prior_probs是一个字典，其中键是类别标签，值是该类别在数据集中出现的概率
prior_probs = {cls: sum(1 for data in train_data if data['Y'] == cls) / num_samples for cls in classes}

# 计算每个特征在每个类别下的条件概率
# 使用defaultdict来简化字典的创建，默认值为一个空字典
conditional_probs = defaultdict(lambda: defaultdict(dict))

# 遍历数据集
for data in train_data:
    # 遍历特征A, B, C
    for feature in ['A', 'B', 'C']:
        # 类别标签
        cls = data['Y']
        # 特征值
        value = data[feature]
        # 增加特征值在对应类别下的计数
        conditional_probs[feature][cls][value] = conditional_probs[feature][cls].get(value, 0) + 1

    # 应用拉普拉斯平滑 (k=1)
# 遍历特征A, B, C
for feature in ['A', 'B', 'C']:
    # 遍历类别标签
    for cls in classes:
        # 计算当前特征在当前类别下的总计数
        total_count = sum(conditional_probs[feature][cls].values())
        # 遍历特征值的可能取值（这里假设是0和1）
        for value in [0, 1]:
            # 如果特征值在当前类别下没有出现过，则初始化为1（拉普拉斯平滑）
            if value not in conditional_probs[feature][cls]:
                conditional_probs[feature][cls][value] = 1
                # 应用拉普拉斯平滑，并计算条件概率
            conditional_probs[feature][cls][value] = (conditional_probs[feature][cls][value] + 1) / (
                        total_count + 2)  # 2是因为有0和1两个值


# 对新样本进行分类的函数
def predict(new_sample):
    # likelihoods是一个字典，用于存储每个类别的似然概率
    likelihoods = {}
    # 遍历类别标签
    for cls in classes:
        # 似然概率初始化为先验概率
        likelihood = prior_probs[cls]
        # 遍历样本中的特征和值
        for feature, value in new_sample.items():
            # 如果特征不是类别标签Y
            if feature != 'Y':
                # 乘以该特征在当前类别下的条件概率
                likelihood *= conditional_probs[feature][cls][value]
                # 将似然概率存储在likelihoods字典中
        likelihoods[cls] = likelihood
        # 返回似然概率最大的类别标签
    return max(likelihoods, key=likelihoods.get)


# 示例新样本
new_sample = {'A': 0, 'B': 1, 'C': 0, 'Y': None}  # 注意：新样本中的'Y'是未知的，通常设置为None或省略

# 进行预测
prediction = predict(new_sample)
print(f"Prediction for {new_sample} is: {prediction}")