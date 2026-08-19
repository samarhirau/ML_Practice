import pandas as pd


df = pd.read_csv("day4\\student_dataset.csv")

print(df.head())

df_drop = df.drop(columns=["Student_Names"])
print(df_drop.head())