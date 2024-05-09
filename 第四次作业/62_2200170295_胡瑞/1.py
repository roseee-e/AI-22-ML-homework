from sklearn.naive_bayes import MultinomialNB
import numpy as np

# 训练数据
X_train = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 1],
                    [1, 0, 1],
                    [0, 0, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 1, 0]])

y_train = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])

# 创建朴素贝叶斯分类器对象
classifier = MultinomialNB()

# 拟合模型
classifier.fit(X_train, y_train)

# 新的输入样本
X_new = np.array([[0, 0, 1]])

# 预测结果
prediction = classifier.predict(X_new)
print("未采用拉普拉斯平滑预测结果:", prediction)

# 使用拉普拉斯平滑（Lap1）重新训练模型
classifier_smooth = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
classifier_smooth.fit(X_train, y_train)

# 预测结果（采用拉普拉斯平滑）
prediction_smooth = classifier_smooth.predict(X_new)
print("采用拉普拉斯平滑（Lap1）预测结果:", prediction_smooth)
