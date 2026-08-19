# Day 49 - Naive Bayes
# SMS Spam Classification

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load Dataset

df = pd.read_csv("Datasets/spam.csv", encoding="latin-1")

print(df.head())
print(df.shape)
print(df.columns)


# 2. Keep Required Columns

df = df.iloc[:, :2]
df.columns = ["label", "message"]

print(df.head())
print(df["label"].value_counts())


# 3. Convert Labels
# ham  -> 0
# spam -> 1

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# 4. Features & Target

X = df["message"]
y = df["label"]


# 5. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training:", len(X_train))
print("Testing :", len(X_test))


# 6. TF-IDF

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Train shape:", X_train_tfidf.shape)
print("Test shape :", X_test_tfidf.shape)


# 7. Multinomial Naive Bayes

mnb = MultinomialNB()

mnb.fit(X_train_tfidf, y_train)

y_pred_mnb = mnb.predict(X_test_tfidf)


# 8. Evaluation

print("\n===== Multinomial NB =====")

print("Accuracy :", accuracy_score(y_test, y_pred_mnb))
print("Precision:", precision_score(y_test, y_pred_mnb))
print("Recall   :", recall_score(y_test, y_pred_mnb))
print("F1 Score :", f1_score(y_test, y_pred_mnb))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_mnb,
    target_names=["Ham", "Spam"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_mnb))


# 9. Bernoulli Naive Bayes

bnb = BernoulliNB()

bnb.fit(X_train_tfidf, y_train)

y_pred_bnb = bnb.predict(X_test_tfidf)

print("\n===== Bernoulli NB =====")

print("Accuracy :", accuracy_score(y_test, y_pred_bnb))
print("Precision:", precision_score(y_test, y_pred_bnb))
print("Recall   :", recall_score(y_test, y_pred_bnb))
print("F1 Score :", f1_score(y_test, y_pred_bnb))


# 10. Test Your Own Messages

messages = [
    "Congratulations! You won a free lottery ticket. Call now!",
    "Hey bro, are you coming to college today?",
    "URGENT! You have won 1000 dollars. Claim your prize now!",
    "Can you send me the notes?"
]

messages_tfidf = vectorizer.transform(messages)

predictions = mnb.predict(messages_tfidf)
probabilities = mnb.predict_proba(messages_tfidf)

for msg, pred, prob in zip(messages, predictions, probabilities):

    label = "SPAM" if pred == 1 else "HAM"

    print("\nMessage:", msg)
    print("Prediction:", label)
    print("Spam Probability:", round(prob[1] * 100, 2), "%")