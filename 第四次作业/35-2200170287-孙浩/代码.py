
import numpy as np

# 训练数据集
X = np.array([[1, 1, 0],
              [0, 1, 0],
              [0, 1, 0],
              [1, 0, 0],
              [0, 1, 1],
              [1, 0, 1],
              [0, 0, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 1, 0]])
y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])

# Laplace 平滑参数
k = 1

# 先验概率
prior_y0 = np.sum(y == 0) / len(y)
prior_y1 = np.sum(y == 1) / len(y)

# 计算条件概率
def compute_conditional_prob(X, y, feature_idx, value, target_value):
    total_count = np.sum(y == target_value)
    feature_count = np.sum((X[:, feature_idx] == value) & (y == target_value))
    return (feature_count + k) / (total_count + k * 2)

# 针对新样本 A=0, B=0, C=1 进行预测
sample = np.array([0, 0, 1])

# 不采用 Laplace 平滑
posterior_y0 = prior_y0
posterior_y1 = prior_y1

for i in range(len(sample)):
    posterior_y0 *= compute_conditional_prob(X, y, i, sample[i], 0)
    posterior_y1 *= compute_conditional_prob(X, y, i, sample[i], 1)

prediction_no_laplace = 1 if posterior_y1 > posterior_y0 else 0

# 使用 Laplace 平滑
posterior_y0_smooth = prior_y0
posterior_y1_smooth = prior_y1

for i in range(len(sample)):
    posterior_y0_smooth *= compute_conditional_prob(X, y, i, sample[i], 0)
    posterior_y1_smooth *= compute_conditional_prob(X, y, i, sample[i], 1)

prediction_with_laplace = 1 if posterior_y1_smooth > posterior_y0_smooth else 0

print("不采用 Laplace 平滑的预测结果：", prediction_no_laplace)
print("采用 Laplace 平滑的预测结果：", prediction_with_laplace)

# 训练数据集
X = np.array([    ['Sunny', 'Hot', 'High', 'Weak'],
    ['Sunny', 'Hot', 'High', 'Strong'],
    ['Overcast', 'Hot', 'High', 'Weak'],
    ['Rain', 'Mild', 'High', 'Weak'],
    ['Rain', 'Cool', 'Normal', 'Weak'],
    ['Rain', 'Cool', 'Normal', 'Strong'],
    ['Overcast', 'Cool', 'Normal', 'Strong'],
    ['Sunny', 'Mild', 'High', 'Weak'],
    ['Sunny', 'Cool', 'Normal', 'Weak'],
    ['Rain', 'Mild', 'Normal', 'Weak'],
    ['Sunny', 'Mild', 'Normal', 'Strong'],
    ['Overcast', 'Mild', 'High', 'Strong'],
    ['Overcast', 'Hot', 'Normal', 'Weak'],
    ['Rain', 'Mild', 'High', 'Strong']
])

y = np.array(['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No'])

# 创建朴素贝叶斯分类器
clf = CategoricalNB()
clf.fit(X, y)

# 测试数据集
test_sample = np.array([['Sunny','Cool','High','Strong']])

# 预测结果
prediction = clf.predict(test_sample)
print("朴素贝叶斯分类器预测的结果为:", prediction)
