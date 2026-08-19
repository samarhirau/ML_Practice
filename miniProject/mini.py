# “I used Random Forest with controlled depth to avoid overfitting and achieved 93% accuracy. I validated the model using confusion matrix and feature importance.”


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier



df = pd.read_csv('miniProject/student-por.csv')

# Data Cleaning: 
# data is clean, no missing values

# Useful numeric features
df = df[["studytime", "failures", "absences", "G1" , "G2" , "G3"]]

# Target column: Pass / Fail
df["Result"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
df.drop("G3", axis=1, inplace=True)

# print(df.head())
# Features & Target
X = df.drop("Result", axis=1)
y = df["Result"]


# Scaling Features 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)




rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=2,
    random_state=42
)

rf.fit(X_train, y_train)



y_pred = rf.predict(X_test)


print("Accuracy:", f"{accuracy_score(y_test, y_pred)*100:.2f} %")

print("Train Accuracy:", rf.score(X_train, y_train))
print("Test Accuracy:", rf.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred))


# imp_df = pd.DataFrame({
#     "Feature": X.columns,
#     "Importance": rf.feature_importances_
# }).sort_values(by="Importance", ascending=False)

# print(imp_df.head(10))