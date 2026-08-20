# Logistic Regression.

# import libraries

import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
# Load dataset

df = pd.read_csv("Datasets/Titanic-Dataset.csv")

# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# drop unnecessary columns
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)


# SELECT FEATURES AND TARGET
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

#  TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
    )


# PREPROCESSING PIPELINE

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler()
            ),
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


# MODEL PIPELINE

model_pipeline = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter=1000)
)

# FIT MODEL
model_pipeline.fit(X_train, y_train)

# PREDICTIONS
y_prob = model_pipeline.predict_proba(X_test)[:, 1]


# EVALUATION
# Confusion matrix
# Accuracy
# Precision
# Recall
# F1-score
for threshold in [0.3, 0.5, 0.7]:
    y_pred_threshold = (y_prob >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_threshold))
    print("Accuracy:", accuracy_score(y_test, y_pred_threshold))
    print("Precision:", precision_score(y_test, y_pred_threshold))
    print("Recall:", recall_score(y_test, y_pred_threshold))
    print("F1 Score:", f1_score(y_test, y_pred_threshold))

