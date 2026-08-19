# Day 47 - Decision Boundaries
# KNN vs Logistic Regression on Iris Dataset

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 1. Load dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Features:", iris.feature_names)
print("Classes:", iris.target_names)
print("Dataset shape:", X.shape)


# 2. Use only 2 features for 2D decision boundary
# Sepal length + Sepal width
X_2d = X[:, :2]

X_train, X_test, y_train, y_test = train_test_split(
    X_2d,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 4. Train KNN models
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(X_train_scaled, y_train)

knn_15 = KNeighborsClassifier(n_neighbors=15)
knn_15.fit(X_train_scaled, y_train)


# 5. Train Logistic Regression
logistic = LogisticRegression(max_iter=1000)
logistic.fit(X_train_scaled, y_train)


# 6. Accuracy
print("\nModel Accuracy:")

print(
    "KNN (k=3):",
    accuracy_score(y_test, knn_3.predict(X_test_scaled))
)

print(
    "KNN (k=15):",
    accuracy_score(y_test, knn_15.predict(X_test_scaled))
)

print(
    "Logistic Regression:",
    accuracy_score(y_test, logistic.predict(X_test_scaled))
)


# 7. Function to plot decision boundary
def plot_decision_boundary(model, X, y, title):

    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1

    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    predictions = model.predict(grid)

    predictions = predictions.reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    plt.contourf(
        xx,
        yy,
        predictions,
        alpha=0.25
    )

    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolor="k",
        s=50
    )

    plt.xlabel("Sepal Length (scaled)")
    plt.ylabel("Sepal Width (scaled)")
    plt.title(title)

    plt.colorbar(scatter, ticks=[0, 1, 2])

    plt.show()


# 8. Plot KNN k=3
plot_decision_boundary(
    knn_3,
    X_train_scaled,
    y_train,
    "KNN Decision Boundary (k=3)"
)


# 9. Plot KNN k=15
plot_decision_boundary(
    knn_15,
    X_train_scaled,
    y_train,
    "KNN Decision Boundary (k=15)"
)


# 10. Plot Logistic Regression
plot_decision_boundary(
    logistic,
    X_train_scaled,
    y_train,
    "Logistic Regression Decision Boundary"
)