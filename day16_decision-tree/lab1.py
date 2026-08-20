import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

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

# Features and target
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

# Test multiple tree depths
for depth in [2, 5, 10]:
    model_pipeline = make_pipeline(
        preprocessor,
        DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )
    )

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)

    print(f"\nMaximum Depth: {depth}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))


# Maximum depth 2 ka Decision Tree simple tha aur usne highest precision achieve ki, lekin uska recall bahut low tha. Maximum depth 5 par recall aur F1-score improve hue. Maximum depth 10 ne highest accuracy, recall aur F1-score achieve kiya, isliye given test set par yeh best model raha. However, deeper trees overfit kar sakte hain, isliye training performance ya cross-validation ke through further verification zaroori hai.