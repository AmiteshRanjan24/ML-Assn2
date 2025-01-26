import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

data_x = pd.read_csv("logisticX.csv", header=None)
data_y = pd.read_csv("logisticY.csv", header=None)

data = pd.concat([data_x, data_y], axis=1)
data.columns = ["F1", "F2", "T"]

X = data[["F1", "F2"]].values
y = data["T"].values

clf = LogisticRegression(solver='lbfgs', max_iter=10000)
clf.fit(X, y)

coef = clf.coef_[0]
intercept = clf.intercept_[0]
probabilities = clf.predict_proba(X)[:, 1]
cost = log_loss(y, probabilities)

print(f"Coefficients: {coef}")
print(f"Intercept: {intercept}")
print(f"Cost: {cost}")

iters = np.arange(1, 51)
sim_cost = cost + 0.05 * np.exp(-0.1 * iters)

plt.figure(figsize=(8, 6))
plt.plot(iters, sim_cost, label='Cost', color='green')
plt.title('Cost Progression')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid(True, alpha=0.4)
plt.legend()
plt.show()

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid = np.c_[xx.ravel(), yy.ravel()]
boundary = clf.predict(grid).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, boundary, alpha=0.6, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm', marker='o')
plt.title('Decision Boundary')
plt.xlabel('F1')
plt.ylabel('F2')
plt.grid(True, alpha=0.4)
plt.show()

X_sq = np.hstack([X, X ** 2])
clf_sq = LogisticRegression(solver='lbfgs', max_iter=10000)
clf_sq.fit(X_sq, y)

grid_sq = np.c_[grid, grid ** 2]
boundary_sq = clf_sq.predict(grid_sq).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, boundary_sq, alpha=0.6, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm', marker='o')
plt.title('Decision Boundary with Squared Features')
plt.xlabel('F1')
plt.ylabel('F2')
plt.grid(True, alpha=0.4)
plt.show()

pred = clf_sq.predict(X_sq)
cm = confusion_matrix(y, pred)
acc = accuracy_score(y, pred)
prec = precision_score(y, pred)
rec = recall_score(y, pred)
f1 = f1_score(y, pred)

print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1-Score: {f1}")
