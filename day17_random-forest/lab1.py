import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv("Datasets/Titanic-Dataset.csv")

# Drop unnecessary columns
df = df.drop(
    columns=["PassengerId", "Name", "Ticket", "Cabin"]
)

# Features
numerical_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]

X = df[numerical_features + categorical_features]
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            numerical_features
        ),
        (
            "cat",
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore")
            ),
            categorical_features
        )
    ]
)

# Random Forest pipeline
model_pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        random_state=42
    )
)

# Train
model_pipeline.fit(X_train, y_train)

# Predict
y_pred = model_pipeline.predict(X_test)

# Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))



# Increasing the number of trees from 100 to 150 slightly improved the model's performance. Accuracy increased from 81.56% to 82.12%, while F1-score increased from 74.02% to 75%. Recall also improved because the model reduced false negatives from 22 to 21. However, the improvement was modest, suggesting that adding more trees may provide only limited additional benefit.