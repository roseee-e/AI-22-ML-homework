import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("D:\\LIUXINYUE\\regress_data1.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
X_train = np.insert(X_train, 0, 1, axis=1)
X_test = np.insert(X_test, 0, 1, axis=1)
m_train, n = X_train.shape
theta = np.zeros(n)
learning_rate = 0.01
iterations = 1000
lambda_regularization = 0.1
train_losses = []
test_losses = []
for iteration in range(iterations):
    predictions_train = np.dot(X_train, theta)
    error_train = predictions_train - y_train
    regularization_term = lambda_regularization * theta
    gradient_train = np.dot(X_train.T, error_train) / m_train + regularization_term
    theta = theta - learning_rate * gradient_train
    train_loss = np.sum(error_train ** 2) / (2 * m_train) + (lambda_regularization / 2) * np.sum(theta ** 2)
    train_losses.append(train_loss)
    predictions_test = np.dot(X_test, theta)
    error_test = predictions_test - y_test
    test_loss = np.sum(error_test ** 2) / (2 * (len(y_test)))
    test_losses.append(test_loss)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training and Test Loss over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()