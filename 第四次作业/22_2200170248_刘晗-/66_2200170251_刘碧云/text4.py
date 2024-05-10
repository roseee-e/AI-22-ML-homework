# 数据准备和朴素贝叶斯分类器的实现
from collections import defaultdict
import numpy as np

# 第一部分: 使用二值特征的朴素贝叶斯分类器

# 假设的数据和标签
features = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0],
                     [0, 1, 1], [1, 0, 1], [0, 0, 0], [0, 1, 1],
                     [1, 0, 0], [1, 1, 0]])
labels = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])


# 朴素贝叶斯分类器实现
class NaiveBayesBinary:
    def __init__(self):
        self.probabilities = {}

    def fit(self, X, y, laplace=1):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # 初始化计数和概率字典
        self.counts = {c: defaultdict(int) for c in self.classes}
        self.probabilities = {c: defaultdict(float) for c in self.classes}
        self.class_prob = {c: (np.sum(y == c) + laplace) / (n_samples + n_classes * laplace) for c in self.classes}

        # 计算特征的条件概率
        for c in self.classes:
            subset = X[y == c]
            total = len(subset)
            for j in range(n_features):
                feature_count = np.sum(subset[:, j])
                self.probabilities[c][j] = (feature_count + laplace) / (total + 2 * laplace)

    def predict(self, X):
        results = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                posterior = np.log(self.class_prob[c])
                for j, val in enumerate(x):
                    prob = self.probabilities[c][j]
                    if val == 1:
                        posterior += np.log(prob)
                    else:
                        posterior += np.log(1 - prob)
                posteriors[c] = posterior
            results.append(max(posteriors, key=posteriors.get))
        return results


# 训练分类器并预测新样本
model = NaiveBayesBinary()
model.fit(features, labels)
prediction = model.predict([[0, 0, 1]])
print("预测结果 A=0, B=0, C=1:", prediction)
# 给定训练数据集
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

# 定义特征名和目标变量
features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
target = 'PlayTennis'

# 从数据中提取特征和标签
X = [row[:-1] for row in data]
y = [row[-1] for row in data]

# 将分类变量编码为数值形式
# 为每个分类变量映射一个整数
feature_values = defaultdict(lambda: defaultdict(int))
label_values = {'Yes': 1, 'No': 0}
reverse_label_values = {v: k for k, v in label_values.items()}

# 为每个特征构建映射
for featureset in X:
    for i, value in enumerate(featureset):
        feature_key = features[i]
        if value not in feature_values[feature_key]:
            feature_values[feature_key][value] = len(feature_values[feature_key])


# 函数来编码数据集
def encode_features(feature_data, feature_maps):
    encoded_data = []
    for featureset in feature_data:
        encoded_featureset = [feature_maps[features[i]][value] for i, value in enumerate(featureset)]
        encoded_data.append(encoded_featureset)
    return encoded_data


# 编码训练特征
encoded_X = encode_features(X, feature_values)


# 定义带有拉普拉斯平滑的朴素贝叶斯分类器
class NaiveBayesClassifier:
    def __init__(self, laplace=1):
        self.laplace = laplace
        self.features_prob = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.class_prob = defaultdict(float)
        self.feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.class_counts = defaultdict(int)
        self.feature_maps = feature_values

    def fit(self, X, y):
        # 计算类别频率和给定类别的特征频率
        for featureset, label in zip(X, y):
            label_num = label_values[label]  # Convert label to numeric
            self.class_counts[label_num] += 1
            for i, feature_value in enumerate(featureset):
                self.feature_counts[i][feature_value][label_num] += 1

        # 计算每个类别的概率
        total_samples = sum(self.class_counts.values())
        for label, count in self.class_counts.items():
            self.class_prob[label] = (count + self.laplace) / (total_samples + self.laplace * len(self.class_counts))

        # 计算每个特征给定类别的概率
        for i, feature_map in self.feature_counts.items():
            for feature_value, label_map in feature_map.items():
                for label, count in label_map.items():
                    self.features_prob[i][feature_value][label] = \
                        (count + self.laplace) / (self.class_counts[label] + self.laplace * len(feature_map))

    def predict(self, featureset):
        # 计算给定特征集的每个类别的概率
        class_scores = {label: np.log(self.class_prob[label]) for label in self.class_counts}
        for label in class_scores:
            for i, feature_value in enumerate(featureset):
                if feature_value in self.features_prob[i]:
                    class_scores[label] += np.log(self.features_prob[i][feature_value].get(label, self.laplace / (
                                self.class_counts[label] + self.laplace * len(self.feature_counts[i]))))
                else:
                    class_scores[label] += np.log(
                        self.laplace / (self.class_counts[label] + self.laplace * len(self.feature_counts[i])))

        # 选择概率最高的类别
        predicted_label_num = max(class_scores, key=class_scores.get)

        return reverse_label_values[predicted_label_num]  # Convert numeric label back to string


# 创建和训练朴素贝叶斯模型
nb_classifier = NaiveBayesClassifier(laplace=1)
nb_classifier.fit(encoded_X, y)

# 测试数据集的特征
test_features = ['Sunny', 'Cool', 'High', 'Strong']
encoded_test_features = encode_features([test_features], feature_values)[0]

# 模型进行预测
predicted_play_tennis = nb_classifier.predict(encoded_test_features)
print("Predicted to play tennis:", predicted_play_tennis)