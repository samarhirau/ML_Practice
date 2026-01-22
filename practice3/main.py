import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load the dataset
df = pd.read_csv('practice3/data/ChocolateSales.csv')

# Clean the dataset
df['Amount'] = (
    df['Amount']
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .astype(float)
)

# DATE column to datetime type
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df = df.drop('Date', axis=1)

# Features and target
X = df.drop('Boxes Shipped', axis=1)
y = df['Boxes Shipped']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Numeric & categorical features
num_features = ['Amount', 'Year', 'Month', 'Day']
cat_features = ['Sales Person', 'Country', 'Product']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

# Full pipeline with RandomForest
model = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Get categorical feature names after one-hot encoding
cat_feature_names = model.named_steps['preprocessing']\
    .named_transformers_['cat'].get_feature_names_out(cat_features)

# Combine all feature names
all_feature_names = list(num_features) + list(cat_feature_names)

# Feature importance from RandomForest
importances = model.named_steps['regressor'].feature_importances_

# Create importance DataFrame
importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)
