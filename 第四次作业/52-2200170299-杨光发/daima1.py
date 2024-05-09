import pandas as pd
from collections import defaultdict

# 训练数据集
data = {
    'A': [1, 0, 0, 1, 0, 1, 0, 0, 1, 1],
    'B': [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    'C': [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
    'y': [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# 计算每个类别的先验概率 P(y)
prior_prob = df['y'].value_counts(normalize=True).to_dict()

# 初始化条件概率字典
conditional_prob = defaultdict(dict)

# 计算条件概率 P(X|y)
for feature in ['A', 'B', 'C']:
    for class_value in df['y'].unique():
        conditional_prob[feature][class_value] = df[df['y'] == class_value][feature].value_counts(normalize=True).get(1, 0)

# 新样本的特征
new_sample = {'A': 0, 'B': 0, 'C': 1}

# 计算后验概率 P(y|X)
posterior_prob = {}
for class_value, prior in prior_prob.items():
    likelihood = 1
    for feature, value in new_sample.items():
        likelihood *= conditional_prob[feature][class_value] if value == 1 else (1 - conditional_prob[feature][class_value])
    posterior_prob[class_value] = prior * likelihood

# 预测结果
predicted_class = max(posterior_prob, key=posterior_prob.get)
print("朴素贝叶斯预测结果:", predicted_class)

# 初始化条件概率字典，拉普拉斯平滑
conditional_prob_laplace = defaultdict(dict)

# 计算条件概率 P(X|y)，使用拉普拉斯平滑
for feature in ['A', 'B', 'C']:
    for class_value in df['y'].unique():
        numerator = df[df['y'] == class_value][feature].sum() + 1
        denominator = len(df[df['y'] == class_value][feature]) + 2  # 二元特征，可能取值为0或1
        conditional_prob_laplace[feature][class_value] = numerator / denominator

# 计算后验概率 P(y|X)，使用拉普拉斯平滑
posterior_prob_laplace = {}
for class_value, prior in prior_prob.items():
    likelihood = 1
    for feature, value in new_sample.items():
        likelihood *= conditional_prob_laplace[feature][class_value] if value == 1 else (1 - conditional_prob_laplace[feature][class_value])
    posterior_prob_laplace[class_value] = prior * likelihood

# 预测结果，使用拉普拉斯平滑
predicted_class_laplace = max(posterior_prob_laplace, key=posterior_prob_laplace.get)
print("拉普拉斯平滑预测结果:", predicted_class_laplace)
