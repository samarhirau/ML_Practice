import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load data
df = pd.read_csv("data/dataset.csv")
df.drop_duplicates(inplace=True)

# Encode categorical columns safely
cat_cols = ['Gender', 'Degree', 'Branch', 'Placement_Status']
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split features and target
X = df.drop('Placement_Status', axis=1)
y = df['Placement_Status']

# Drop low-importance / leakage-prone features
drop_cols = [
    'Student_ID',
    'Age',
    'Gender',
    'Degree',
    'Branch',
    'Internships',
    'Soft_Skills_Rating',
    'Certifications'
]

X = X.drop(drop_cols, axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)
