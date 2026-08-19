# ============================================================
# POLYNOMIAL REGRESSION - HOUSING PRICE PREDICTION
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

sns.set_theme(style="whitegrid")
np.random.seed(42)


# ============================================================
# 1. LOAD DATA
# ============================================================

CSV_PATH = Path("Datasets/Housing.csv")

df = pd.read_csv(CSV_PATH)

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. BASIC DATA INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFO")
print("=" * 60)

print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================

TARGET = "price"

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:", TARGET)


# ============================================================
# 4. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 5. CHECK NUMERICAL CORRELATION WITH PRICE
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION WITH PRICE")
print("=" * 60)

correlation = (
    df[numerical_features + [TARGET]]
    .corr()[TARGET]
    .sort_values(ascending=False)
)

print(correlation)


# ============================================================
# 6. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10, 7))

sns.heatmap(
    df[numerical_features + [TARGET]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ============================================================
# 7. CATEGORICAL FEATURES VS PRICE
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL FEATURE ANALYSIS")
print("=" * 60)

for feature in categorical_features:

    print(f"\n--- {feature} ---")

    print(
        df.groupby(feature)[TARGET]
        .agg(["mean", "median", "count"])
        .sort_values("mean", ascending=False)
    )


# ============================================================
# 8. VISUALIZE NUMERICAL FEATURES VS PRICE
# ============================================================

for feature in numerical_features:

    plt.figure(figsize=(7, 5))

    sns.scatterplot(
        data=df,
        x=feature,
        y=TARGET,
        alpha=0.6
    )

    plt.title(f"{feature} vs {TARGET}")
    plt.tight_layout()
    plt.show()


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 10. PREPROCESSING
# ============================================================

# Numerical features:
# PolynomialFeatures + StandardScaler
#
# Categorical features:
# OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                (
                    "poly",
                    PolynomialFeatures(
                        degree=2,
                        include_bias=False
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]),
            numerical_features
        ),

        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 11. CREATE POLYNOMIAL REGRESSION MODEL
# ============================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


# ============================================================
# 12. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 13. PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 14. EVALUATION
# ============================================================

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print("R² Score :", round(r2, 4))
print("MAE      :", round(mae, 2))
print("RMSE     :", round(rmse, 2))


# ============================================================
# 15. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

# Perfect prediction line
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")

plt.tight_layout()
plt.show()


# ============================================================
# 16. RESIDUAL ANALYSIS
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted Price")

plt.tight_layout()
plt.show()


# ============================================================
# 17. COMPARE POLYNOMIAL DEGREES
# ============================================================

degrees = [1, 2, 3]

results = {}

for degree in degrees:

    print(f"\nTraining Polynomial Degree {degree}...")

    current_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    (
                        "poly",
                        PolynomialFeatures(
                            degree=degree,
                            include_bias=False
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]),
                numerical_features
            ),

            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    current_model = Pipeline([
        (
            "preprocessor",
            current_preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ])

    current_model.fit(
        X_train,
        y_train
    )

    predictions = current_model.predict(X_test)

    results[degree] = {
        "model": current_model,
        "r2": r2_score(
            y_test,
            predictions
        ),
        "mae": mean_absolute_error(
            y_test,
            predictions
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )
    }


# ============================================================
# 18. RESULTS TABLE
# ============================================================

metrics = pd.DataFrame({
    degree: {
        "R2": results[degree]["r2"],
        "MAE": results[degree]["mae"],
        "RMSE": results[degree]["rmse"]
    }
    for degree in degrees
}).T

metrics.index.name = "Polynomial Degree"

print("\n" + "=" * 60)
print("POLYNOMIAL DEGREE COMPARISON")
print("=" * 60)

print(
    metrics.sort_values(
        "RMSE"
    )
)


# ============================================================
# 19. SELECT BEST DEGREE
# ============================================================

best_degree = min(
    results,
    key=lambda degree: results[degree]["rmse"]
)

best_model = results[best_degree]["model"]

best_predictions = best_model.predict(X_test)

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Best Degree:", best_degree)

print(
    "R²:",
    round(
        r2_score(
            y_test,
            best_predictions
        ),
        4
    )
)

print(
    "MAE:",
    round(
        mean_absolute_error(
            y_test,
            best_predictions
        ),
        2
    )
)

print(
    "RMSE:",
    round(
        np.sqrt(
            mean_squared_error(
                y_test,
                best_predictions
            )
        ),
        2
    )
)


# ============================================================
# 20. FEATURE IMPORTANCE / COEFFICIENT ANALYSIS
# ============================================================

# Get transformed feature names

preprocessor_fitted = best_model.named_steps[
    "preprocessor"
]

regressor = best_model.named_steps[
    "regressor"
]

feature_names = (
    preprocessor_fitted
    .get_feature_names_out()
)

coefficients = regressor.coef_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "absolute_coefficient": np.abs(coefficients)
})

feature_importance = feature_importance.sort_values(
    "absolute_coefficient",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP IMPORTANT TRANSFORMED FEATURES")
print("=" * 60)

print(
    feature_importance.head(20)
)


# ============================================================
# 21. PLOT TOP FEATURES
# ============================================================

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_features,
    x="absolute_coefficient",
    y="feature"
)

plt.title(
    "Top Features by Absolute Model Coefficient"
)

plt.xlabel("Absolute Coefficient")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("Dataset Shape:", df.shape)

print("Numerical Features:", numerical_features)

print("Categorical Features:", categorical_features)

print("Best Polynomial Degree:", best_degree)

print(
    "Best R²:",
    round(
        results[best_degree]["r2"],
        4
    )
)

print(
    "Best MAE:",
    round(
        results[best_degree]["mae"],
        2
    )
)

print(
    "Best RMSE:",
    round(
        results[best_degree]["rmse"],
        2
    )
)