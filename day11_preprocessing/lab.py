

# # Titanic Dataset — Exploratory Data Analysis


# ## 1. Introduction

# ## 2. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# ## 3. Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# ## 4. Initial Inspection
print("Dataset Shape:", df.shape)
print("\nFirst Few Rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())

# ## 5. Data Cleaning
# Remove duplicates
df = df.drop_duplicates()
print(f"\nDataset shape after removing duplicates: {df.shape}")

# Handle missing values
print("\nHandling Missing Values...")

# Fill Age with median
age_imputer = SimpleImputer(strategy='median')
df['Age'] = age_imputer.fit_transform(df[['Age']])

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing values)
df.drop('Cabin', axis=1, inplace=True)

# Drop PassengerId and Ticket (not useful for prediction)
df.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)

print(f"Missing values after cleaning:\n{df.isnull().sum()}")

# ## 6. Feature Engineering
print("\nFeature Engineering...")

# Create FamilySize feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Create IsAlone feature
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Create AgeGroup feature
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 20, 35, 60, 100], 
                        labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

# Encode categorical variables
le_sex = LabelEncoder()
df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])

le_embarked = LabelEncoder()
df['Embarked_encoded'] = le_embarked.fit_transform(df['Embarked'])

le_agegroup = LabelEncoder()
df['AgeGroup_encoded'] = le_agegroup.fit_transform(df['AgeGroup'].astype(str))

print("New features created:")
print(df[['FamilySize', 'IsAlone', 'AgeGroup', 'Sex_encoded']].head())

# ## 7. Exploratory Data Analysis
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ### Age Distribution
axes[0, 0].hist(df['Age'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].set_title('Age Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# ### Survival by Passenger Class
survival_class = df.groupby('Pclass')['Survived'].mean()
axes[0, 1].bar(survival_class.index, survival_class.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0, 1].set_title('Survival Rate by Passenger Class', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Passenger Class')
axes[0, 1].set_ylabel('Survival Rate')
axes[0, 1].set_ylim([0, 1])

# ### Age vs Fare
axes[1, 0].scatter(df['Age'], df['Fare'], alpha=0.5, s=30, color='coral')
axes[1, 0].set_title('Age vs Fare', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Fare')

# ### Survival by Sex
survival_sex = df.groupby('Sex')['Survived'].mean()
axes[1, 1].bar(survival_sex.index, survival_sex.values, color=['#FF69B4', '#4169E1'])
axes[1, 1].set_title('Survival Rate by Sex', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Sex')
axes[1, 1].set_ylabel('Survival Rate')
axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('titanic_eda.png', dpi=100, bbox_inches='tight')
print("\nPlot saved as 'titanic_eda.png'")
plt.show()

# Additional: Correlation Heatmap
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation Heatmap - Titanic Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('titanic_correlation.png', dpi=100, bbox_inches='tight')
print("Correlation heatmap saved as 'titanic_correlation.png'")
plt.show()

# Final processed dataset info
print("\nFinal Processed Dataset:")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nProcessed Data Sample:")
print(df.head())

# ## 8. Key Insights

# 1. Missing value handling completed with appropriate strategies
# 2. Feature engineering created meaningful predictive variables
# 3. Visualizations reveal patterns in survival rates across different groups

# ## 9. Conclusion