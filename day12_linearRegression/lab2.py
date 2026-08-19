# 0) Setup

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import joblib, numpy as np
sns.set_theme(style='whitegrid', context='notebook')


# 1) Load data
from pathlib import Path
USE_CALIFORNIA = False
CSV_PATH = Path('Datasets/retai.csv')
TARGET = 'RETAIL SALES'

df = pd.read_csv(CSV_PATH).dropna(subset=[TARGET]).copy()
y = df[TARGET].astype(float)
X = df.drop(columns=[TARGET])
numeric_features = X.select_dtypes(include='number').columns.tolist()
categorical_features = X.select_dtypes(exclude='number').columns.tolist()

# 2) Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3) Preprocessing + Linear Regression pipeline

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

preprocess = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
], remainder='drop')
 
model = Pipeline([
    ('preprocess', preprocess),
    ('reg', LinearRegression())
]).fit(X_train, y_train)

# 4) Evaluate (MAE, RMSE, R²) vs a baseline

y_pred = model.predict(X_test)
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"LinearRegression → R²={r2:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}")
 
baseline = np.full_like(y_test, y_train.mean(), dtype=float)
print("Baseline(mean)  →",
      f"R²={r2_score(y_test, baseline):.3f}",
      f"MAE={mean_absolute_error(y_test, baseline):.3f}",
      f"RMSE={np.sqrt(mean_squared_error(y_test, baseline)):.3f}")

# Visual diagnostics
resid = y_test - y_pred
 
# Predicted vs Actual
plt.figure(); 
plt.scatter(y_test, y_pred, alpha=0.6)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims)
plt.title('Predicted vs Actual'); plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.tight_layout(); plt.show()
 
# Residuals vs Predicted
plt.figure();
plt.scatter(y_pred, resid, alpha=0.6)
plt.axhline(0, linestyle='--')
plt.title('Residuals vs Predicted'); plt.xlabel('Predicted'); plt.ylabel('Residual'); plt.tight_layout(); plt.show()
 
# Residuals histogram
plt.figure(); 
plt.hist(resid, bins=30)
plt.title('Residuals Histogram'); plt.xlabel('Residual'); plt.ylabel('Count'); plt.tight_layout(); plt.show()