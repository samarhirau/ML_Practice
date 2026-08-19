from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



df = pd.read_csv('miniProject/student-por.csv')

# Data Cleaning:
# data is clean, no missing values

# Useful numeric features
df = df[["studytime", "failures", "absences", "G3"]]
# Target column: Pass / Fail
df["Result"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
df.drop("G3", axis=1, inplace=True)

# Features & Target
X = df.drop("Result", axis=1)
y = df["Result"]

# Scaling Features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=2, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)


print("Accuracy Logistic Regression:", acc_lr)
print("Accuracy KNN:", acc_knn)
print("Accuracy Decision Tree:", acc_dt)



models = ['Logistic Regression', 'KNN', 'Decision Tree']
accuracies = [acc_lr, acc_knn, acc_dt]

plt.figure(figsize=(8,5))
plt.bar(models, accuracies, color=['blue','green','orange'])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()
