
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_absolute_error




df = pd.read_csv('practice6/data/car_prices.csv')


df.dropna(subset=['sellingprice'] , inplace=True)

X = df.drop('sellingprice' , axis=1)
y = df['sellingprice']

# preprocessing data 
numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')


X['transmission'] = X['transmission'].fillna('Unknown')

numeric_features = X.select_dtypes(
    include=['int64', 'float64']
).columns.to_list()

categorical_features = X.select_dtypes(
    include=['object']
).columns.to_list()

for col in categorical_features:
    top = X[col].value_counts().nlargest(50).index
    X[col] = X[col].where(X[col].isin(top), 'Other')



numeric_transformer = Pipeline(steps=[
    ('imputer', numeric_imputer),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', categorical_imputer),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)
)
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer,numeric_features),
        ('cat' , categorical_transformer , categorical_features)
    ]
)

model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor' , SGDRegressor(max_iter=1000, tol=1e-3))
    ]
)

X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size=0.2 , random_state=42
)


model.fit(X_train , y_train)

y_pred = model.predict(X_test)




r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R2 Score: {r2}")
print(f"Mean Absolute Error: {mae}")