import pandas as pd

df = pd.read_csv('D:\Machine Learning\day4\employees.csv')

df["First Name"] = df["First Name"].mode()[0]

# convert gender to categorical

df["Gender"] = df["Gender"].astype('category')


df["Senior Management"] = df["Senior Management"].mode()[0]
df["Team"] = df["Team"].mode()[0]


print(df.isna().sum())