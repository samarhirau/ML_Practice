# Polynomial Regression

# 0) Setup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set_theme(style='whitegrid', context='notebook')
np.random.seed(42)

# 1) Make or load 1D data (easy to plot)
USE_CSV = False
CSV_PATH = Path('Datasets/retai.csv')
X_COL, Y_COL = 'YEAR', 'RETAIL SALES'

if USE_CSV:
    df = pd.read_csv(CSV_PATH)
    missing = [col for col in [X_COL, Y_COL] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {CSV_PATH}: {missing}")
    df = df[[X_COL, Y_COL]].dropna()
    X = df[[X_COL]].values.astype(float)
    y = df[Y_COL].values.astype(float)
else:
    rng = np.random.RandomState(42)
    x = np.linspace(-2, 2, 500)
    y = 1.2 * x**3 - 0.8 * x**2 + 3 * x + rng.normal(0, 1.5, size=x.shape)
    X = x.reshape(-1, 1)

# 2) Train–test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 3) Fit models for degrees = 1, 2, 3, 5
degrees = [1, 2, 3, 5]
results = {}
for d in degrees:
    pipe = make_pipeline(PolynomialFeatures(degree=d, include_bias=False), LinearRegression())
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    results[d] = {
        'model': pipe,
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    }

# 4) Visualize fits on one plot
X_plot = np.linspace(X.min() - 0.5, X.max() + 0.5, 400).reshape(-1, 1)
plt.figure(figsize=(7, 5))
plt.scatter(X_train, y_train, alpha=0.4, label='train')
plt.scatter(X_test, y_test, alpha=0.6, label='test')
for d in degrees:
    plt.plot(X_plot, results[d]['model'].predict(X_plot), label=f'degree={d}')
plt.title('Polynomial Regression Fits by Degree')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()

# 5) Metrics table (test set)
metrics = pd.DataFrame({d: {k: v for k, v in results[d].items() if k != 'model'} for d in degrees}).T
metrics.index.name = 'degree'
print(metrics.sort_values('rmse'))

# 6) Residuals vs predicted (spot under/overfit)
choose = [1, 3]
fig, axes = plt.subplots(1, len(choose), figsize=(12, 4))
for ax, d in zip(axes, choose):
    mdl = results[d]['model']
    y_pred = mdl.predict(X_test)
    resid = y_test - y_pred
    ax.scatter(y_pred, resid, alpha=0.6)
    ax.axhline(0, linestyle='--')
    ax.set_title(f'Residuals — degree {d}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residual')
plt.tight_layout()
plt.show()

# 7) (Optional) Reduce overfitting with Ridge
alphas = [0.0, 0.01, 0.1, 1.0, 10.0]
ridge_scores = []
for a in alphas:
    mdl = make_pipeline(PolynomialFeatures(5, include_bias=False), StandardScaler(with_mean=False), Ridge(alpha=a))
    mdl.fit(X_train, y_train)
    yhat = mdl.predict(X_test)
    ridge_scores.append((a, r2_score(y_test, yhat), np.sqrt(mean_squared_error(y_test, yhat))))
print('Ridge scores:', ridge_scores)

# 8) Best degree selection
best_degree = min(results, key=lambda d: results[d]['rmse'])

# 9) Final fit with best_degree
final_model = make_pipeline(PolynomialFeatures(int(best_degree), include_bias=False), LinearRegression()).fit(X_train, y_train)
y_pred_final = final_model.predict(X_test)
print('Best degree:', best_degree)
print('Test R^2:', round(r2_score(y_test, y_pred_final), 3))
print('Test RMSE:', round(np.sqrt(mean_squared_error(y_test, y_pred_final)), 3))
print('Test MAE:', round(mean_absolute_error(y_test, y_pred_final), 3))


