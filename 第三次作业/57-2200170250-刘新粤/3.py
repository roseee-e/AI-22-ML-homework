import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
data = pd.read_csv(r"D:\LIUXINYUE\ex2data1.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def logistic_regression(X_train, y_train, X_val, y_val, C=1.0):
    model = LogisticRegression(C=C)
    model.fit(X_train, y_train)
    train_loss = -np.mean(y_train * np.log(model.predict_proba(X_train)[:, 1]) + (1 - y_train) * np.log(1 - model.predict_proba(X_train)[:, 1]))
    val_loss = -np.mean(y_val * np.log(model.predict_proba(X_val)[:, 1]) + (1 - y_val) * np.log(1 - model.predict_proba(X_val)[:, 1]))
    return model, train_loss, val_loss
def plot_loss_curve(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    return accuracy, precision, recall, f1, auc, fpr, tpr
model, train_losses, val_losses = logistic_regression(X_train, y_train, X_test, y_test)
plot_loss_curve(train_losses, val_losses)
accuracy, precision, recall, f1, auc, fpr, tpr = evaluate_model(model, X_test, y_test)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", auc)
plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()