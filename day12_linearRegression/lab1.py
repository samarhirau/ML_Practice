# Polynomial Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("Datasets/retai.csv")

# print(df.head())
# print(df.info())
# print(df.describe())




age_imputer = SimpleImputer(strategy='median')
df['RETAIL SALES'] = age_imputer.fit_transform(df[['RETAIL SALES']])[:, 0]

df["SUPPLIER"] = df["SUPPLIER"].fillna(df["SUPPLIER"].mode()[0])

X = df[["YEAR", "MONTH", "RETAIL TRANSFERS", "WAREHOUSE SALES"]]
y = df["RETAIL SALES"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("linear", LinearRegression())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"R^2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual RETAIL SALES')
plt.ylabel('Predicted RETAIL SALES')
plt.title('Polynomial Regression: Actual vs Predicted')
plt.grid(True)
plt.savefig('polynomial_regression.png')
plt.show()
