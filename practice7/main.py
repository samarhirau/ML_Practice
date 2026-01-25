

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score

df = pd.read_csv('practice7/data/Customers.csv')

df.drop_duplicates(inplace=True)
df.reset_index(drop=True , inplace=True)

# Drop ID and other non-predictive columns
X = df.drop(['CustomerID', 'Spending Score (1-100)'], axis=1, errors='ignore')
y = df['Spending Score (1-100)']

# preprocessing data
numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

numeric_features = X.select_dtypes(
    include=['int64', 'float64']
).columns.to_list()

categorical_features = X.select_dtypes(
    include=['object']
).columns.to_list()


numeric_transformer = Pipeline(steps=[
    ('imputer', numeric_imputer),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', categorical_imputer),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])


preprocessor = ColumnTransformer(
    transformers=[  
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# create the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])


# train split 

X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size=0.2 , random_state=42
)

# model training
model.fit(X_train , y_train)


# model evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test , y_pred)
r2 = r2_score(y_test , y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}') 

