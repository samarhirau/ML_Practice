import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

df = pd.read_csv('practice2/DATA/car_data.csv')
df = df.dropna().reset_index(drop=True)
df.drop_duplicates(inplace=True)

X = df.drop('Price', axis=1)
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
numeric_features = ['Year', 'Mileage']
numeric_transformer = StandardScaler()

categorical_features = ['Brand', 'Model', 'Fuel Type', 'Transmission' , 'Condition']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')


X_train[numeric_features] = numeric_transformer.fit_transform(
    X_train[numeric_features]
)
X_test[numeric_features] = numeric_transformer.transform(
    X_test[numeric_features]
)
X_train_cat = categorical_transformer.fit_transform(X_train[categorical_features])
X_test_cat = categorical_transformer.transform(X_test[categorical_features])

import numpy as np
X_train_cat_df = pd.DataFrame(
    X_train_cat.toarray(),
    columns=categorical_transformer.get_feature_names_out(categorical_features)
)

X_test_cat_df = pd.DataFrame(
    X_test_cat.toarray(),
    columns=categorical_transformer.get_feature_names_out(categorical_features)
)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

X_train_final = pd.concat(
    [X_train.drop(categorical_features, axis=1), X_train_cat_df], axis=1
, ignore_index=False

)
X_test_final = pd.concat(
    [X_test.drop(categorical_features, axis=1), X_test_cat_df], axis=1, ignore_index=False
)

model.fit(X_train_final, y_train)
y_pred = model.predict(X_test_final)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

importances = permutation_importance(
    model, X_test_final, y_test, n_repeats=10, random_state=42
)
feature_importances = pd.Series(
    importances.importances_mean, index=X_test_final.columns
).sort_values(ascending=False)

print(f'R² Score: {r2:.4f}')
print(f'Mean Absolute Error: {mae:.2f}')

