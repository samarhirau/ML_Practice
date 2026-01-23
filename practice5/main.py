import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# load data
df = pd.read_csv('practice5/data/house_prices_practice.csv')


# split target
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model selection
model = LinearRegression()

# train
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# evaluation
print("R2 Score:", f"{r2_score(y_test, y_pred) * 100:.2f}%")
print("MAE:", mean_absolute_error(y_test, y_pred))

