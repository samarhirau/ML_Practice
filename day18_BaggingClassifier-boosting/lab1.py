from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)



# load dataset

pd = pd.read_csv("Datasets/Titanic-Dataset.csv")

# drop unnecessary columns
pd.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True
)

# select features and target
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

X = pd[numerical_features + categorical_features]
y = pd["Survived"]

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

bagging_model = make_pipeline(
    preprocessor,
    BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=None,
            random_state=42
        ),
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
)

bagging_model.fit(X_train, y_train)

y_pred_bagging = bagging_model.predict(X_test)

print("Bagging Results")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_bagging))
print("Accuracy:", accuracy_score(y_test, y_pred_bagging))
print("Precision:", precision_score(y_test, y_pred_bagging))
print("Recall:", recall_score(y_test, y_pred_bagging))
print("F1 Score:", f1_score(y_test, y_pred_bagging))
