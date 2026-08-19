import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("student_dataset.csv")

# 1. Basic inspection
print(df.shape)
print(df.columns)
print(df.head())
print(df.info())

# 2. Missing values
print(df.isnull().sum())

# 3. Duplicates
print("Duplicates:", df.duplicated().sum())

# 4. Statistics
print(df.describe())

# 5. Correlation
print(df.corr(numeric_only=True))

# 6. Histogram
df.hist(figsize=(10, 8))
plt.show()

# 7. Correlation heatmap
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True
)
plt.show()