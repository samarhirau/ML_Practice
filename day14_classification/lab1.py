# Day 46 - KNN Classification on Iris Dataset

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nTarget classes:")
print(iris.target_names)


# --------------------------------------------------
# 2. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# --------------------------------------------------
# 3. Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 4. Try Different K Values
# --------------------------------------------------

k_values = [1, 3, 5, 7, 9, 11]

results = []

for k in k_values:

    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean"
    )

    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    results.append({
        "K": k,
        "Accuracy": accuracy
    })


results_df = pd.DataFrame(results)

print("\nK Comparison:")
print(results_df)


# --------------------------------------------------
# 5. Find Best K
# --------------------------------------------------

best_k = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "K"
]

best_accuracy = results_df["Accuracy"].max()

print("\nBest K:", best_k)
print("Best Accuracy:", round(best_accuracy, 4))


# --------------------------------------------------
# 6. Train Final Model
# --------------------------------------------------

best_knn = KNeighborsClassifier(
    n_neighbors=best_k,
    metric="euclidean"
)

best_knn.fit(X_train_scaled, y_train)

y_pred = best_knn.predict(X_test_scaled)


# --------------------------------------------------
# 7. Classification Report
# --------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# --------------------------------------------------
# 8. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
).plot()

plt.title(f"KNN Confusion Matrix (K={best_k})")
plt.show()


# --------------------------------------------------
# 9. K vs Accuracy Plot
# --------------------------------------------------

plt.plot(
    results_df["K"],
    results_df["Accuracy"],
    marker="o"
)

plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("KNN: K vs Accuracy")
plt.grid(True)
plt.show()


# --------------------------------------------------
# 10. Compare Distance Metrics
# --------------------------------------------------

distance_results = []

for metric in ["euclidean", "manhattan"]:

    knn = KNeighborsClassifier(
        n_neighbors=best_k,
        metric=metric
    )

    knn.fit(X_train_scaled, y_train)

    y_pred_metric = knn.predict(X_test_scaled)

    accuracy = accuracy_score(
        y_test,
        y_pred_metric
    )

    distance_results.append({
        "Distance Metric": metric,
        "Accuracy": accuracy
    })


distance_df = pd.DataFrame(distance_results)

print("\nDistance Metric Comparison:")
print(distance_df)