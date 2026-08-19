# DAY 50 - BIAS VARIANCE TRADE-OFF
# Polynomial Regression Experiment

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures,  StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# 2. LOAD DATASET

# Apne dataset ke according path change karo
df = pd.read_csv("Datasets/Housing.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# 3. BASIC DATA CHECK

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())


# 4. SELECT FEATURES AND TARGET


numerical_features = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking"
]
categorical_features = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus"
]

X = df[numerical_features + categorical_features]
y = df["price"]

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)




# 5. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# 5. PREPROCESSOR

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_features
        )
    ]
)
# 6. POLYNOMIAL REGRESSION

degrees = [1, 3, 10]

results = []

for degree in degrees:
    

    # Polynomial Regression Pipeline
    model = make_pipeline(
        preprocessor,
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        LinearRegression()
    )

    # Train
    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Training Metrics

    train_r2 = r2_score(y_train, train_pred)

    train_mae = mean_absolute_error(
        y_train,
        train_pred
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_pred
        )
    )

    # Testing Metrics

    test_r2 = r2_score(y_test, test_pred)

    test_mae = mean_absolute_error(
        y_test,
        test_pred
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_pred
        )
    )

    # Store results
    results.append({
        "Degree": degree,

        "Train R²": train_r2,
        "Test R²": test_r2,

        "Train MAE": train_mae,
        "Test MAE": test_mae,

        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse
    })


# 7. RESULTS TABLE

results_df = pd.DataFrame(results)

print("BIAS-VARIANCE RESULTS")

print(
    results_df.round(4).to_string(index=False)
)


# 8. IDENTIFY BEST DEGREE

best_index = results_df["Test R²"].idxmax()

best_degree = results_df.loc[
    best_index,
    "Degree"
]

best_r2 = results_df.loc[
    best_index,
    "Test R²"
]

best_rmse = results_df.loc[
    best_index,
    "Test RMSE"
]

best_mae = results_df.loc[
    best_index,
    "Test MAE"
]

print("BEST MODEL")

print("Best Degree :", best_degree)
print("Test R²     :", round(best_r2, 4))
print("Test RMSE   :", round(best_rmse, 4))
print("Test MAE    :", round(best_mae, 4))


# 9. DETECT BIAS / VARIANCE

print("MODEL ANALYSIS")

for _, row in results_df.iterrows():

    degree = row["Degree"]

    train_r2 = row["Train R²"]
    test_r2 = row["Test R²"]

    gap = train_r2 - test_r2

    print(f"\nDegree {degree}")

    print("Train R²:", round(train_r2, 4))
    print("Test R² :", round(test_r2, 4))
    print("Gap     :", round(gap, 4))

    # High Bias / Underfitting
    if train_r2 < 0.80 and test_r2 < 0.80:

        print("→ Possible HIGH BIAS / UNDERFITTING")

    # High Variance / Overfitting
    elif gap > 0.10:

        print("→ Possible HIGH VARIANCE / OVERFITTING")

    # Good balance
    else:

        print("→ Good Bias-Variance Balance")


# 10. PLOT TRAINING vs TESTING R²

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Degree"],
    results_df["Train R²"],
    marker="o",
    label="Training R²"
)

plt.plot(
    results_df["Degree"],
    results_df["Test R²"],
    marker="o",
    label="Testing R²"
)

plt.xlabel("Polynomial Degree")
plt.ylabel("R² Score")
plt.title("Bias-Variance Trade-off: R²")
plt.legend()
plt.grid(True)

plt.show()


# 11. PLOT TRAINING vs TESTING RMSE

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Degree"],
    results_df["Train RMSE"],
    marker="o",
    label="Training RMSE"
)

plt.plot(
    results_df["Degree"],
    results_df["Test RMSE"],
    marker="o",
    label="Testing RMSE"
)

plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.title("Bias-Variance Trade-off: RMSE")
plt.legend()
plt.grid(True)

plt.show()


# 12. ACTUAL vs PREDICTED FOR BEST MODEL

best_model = make_pipeline(
    preprocessor,
    PolynomialFeatures(degree=int(best_degree)),
    LinearRegression()
)

best_model.fit(X_train, y_train)

best_predictions = best_model.predict(X_test)


plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.7
)

# Perfect prediction line
minimum = min(y_test.min(), best_predictions.min())
maximum = max(y_test.max(), best_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title(
    f"Actual vs Predicted - Degree {int(best_degree)}"
)

plt.grid(True)

plt.show()


# 13. OPTIONAL: RIDGE REGULARIZATION

print("\n===================================")
print("RIDGE REGULARIZATION")
print("===================================")

alphas = [0.01, 0.1, 1, 10, 100]

ridge_results = []

for alpha in alphas:

    ridge_model = make_pipeline(
        preprocessor,
        PolynomialFeatures(degree=int(best_degree)),
        Ridge(alpha=alpha)
    )

    ridge_model.fit(
        X_train,
        y_train
    )

    ridge_pred = ridge_model.predict(X_test)

    ridge_r2 = r2_score(
        y_test,
        ridge_pred
    )

    ridge_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            ridge_pred
        )
    )

    ridge_mae = mean_absolute_error(
        y_test,
        ridge_pred
    )

    ridge_results.append({
        "Alpha": alpha,
        "R²": ridge_r2,
        "RMSE": ridge_rmse,
        "MAE": ridge_mae
    })


ridge_df = pd.DataFrame(ridge_results)

print(
    ridge_df.round(4).to_string(index=False)
)


# 14. FINAL CONCLUSION

print("FINAL CONCLUSION")

print(
    f"""
Best Polynomial Degree: {int(best_degree)}

Test R²   : {best_r2:.4f}
Test RMSE : {best_rmse:.4f}
Test MAE  : {best_mae:.4f}

Interpretation:
- Degree 1 → usually simpler, may have high bias
- Degree 3 → can capture more complex patterns
- Degree 10 → highly flexible and may have high variance
- Best model → the one that generalizes best on unseen test data
"""
)