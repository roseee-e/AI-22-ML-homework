import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def computeCost(X, Y, w):
    P = sigmoid(np.dot(w.T, X))
    loss = np.sum(-Y*np.log(P) - (1-Y)*np.log(1 - P)) / X.shape[1]
    return loss, P

def gradientDescent(w, X, Y, alpha):
    error = sigmoid(np.dot(w.T, X)) - Y
    grad = np.dot(X, error.T) / X.shape[1]
    w -= alpha * grad
    return w

def logisticRegression(X_train, X_val, Y_train, Y_val, alpha, iters):
    feature_dim = X_train.shape[0]
    W = np.zeros((feature_dim, 1))
    loss_train_hist, loss_val_hist = [], []

    for i in range(iters):
        loss_train, _ = computeCost(X_train, Y_train, W)
        loss_val, _ = computeCost(X_val, Y_val, W)
        loss_train_hist.append(loss_train)
        loss_val_hist.append(loss_val)

        W = gradientDescent(W, X_train, Y_train, alpha)

    return loss_train_hist, loss_val_hist, W

def predict(w, X):
    probability = sigmoid(np.dot(w.T, X))
    y_hat = probability >= 0.5
    return probability, y_hat

data = pd.read_csv('ex2data1.txt', header=None)
X = data.iloc[:, :2].values
y = data.iloc[:, 2].values

X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=42)

alpha = 0.001
iters = 1000

loss_train_hist, loss_val_hist, W = logisticRegression(X_train.T, X_val.T, Y_train.T, Y_val.T, alpha, iters)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(iters), loss_train_hist, 'r', label='Training Loss')
ax.plot(np.arange(iters), loss_val_hist, 'b', label='Validation Loss')
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
ax.set_title('Cost vs. Iterations')
plt.legend()
plt.show()

_, y_prob_train = predict(W, X_train.T)
fpr_train, tpr_train, _ = roc_curve(Y_train, y_prob_train.T)

_, y_prob_val = predict(W, X_val.T)
fpr_val, tpr_val, _ = roc_curve(Y_val, y_prob_val.T)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_train, tpr_train, 'b', label='Training ROC Curve')
ax.plot(fpr_val, tpr_val, 'r', label='Validation ROC Curve')
ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
plt.legend()
plt.show()

_, y_hat_train = predict(W, X_train.T)
_, y_hat_val = predict(W, X_val.T)

acc_train = accuracy_score(Y_train, y_hat_train.T)
precision_train = precision_score(Y_train, y_hat_train.T)
recall_train = recall_score(Y_train, y_hat_train.T)
f1_train = f1_score(Y_train, y_hat_train.T)
auc_train = roc_auc_score(Y_train, y_hat_train.T)

acc_val = accuracy_score(Y_val, y_hat_val.T)
precision_val = precision_score(Y_val, y_hat_val.T)
recall_val = recall_score(Y_val, y_hat_val.T)
f1_val = f1_score(Y_val, y_hat_val.T)
auc_val = roc_auc_score(Y_val, y_hat_val.T)

print("Training Set Metrics:")
print("Accuracy:", acc_train)
print("Precision:", precision_train)
print("Recall:", recall_train)
print("F1 Score:", f1_train)
print("AUC:", auc_train)

print("\nValidation Set Metrics:")
print("Accuracy:", acc_val)
print("Precision:", precision_val)
print("Recall:", recall_val)
print("F1 Score:", f1_val)
print("AUC:", auc_val)
