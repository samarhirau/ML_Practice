import pandas as pd


for chunk in pd.read_csv("day4\employees.csv", chunksize=500):
# compute the mean of the "Salary" column for each chunk
    mean_salary = chunk["Salary"].mean()
    print(f"Mean Salary for this chunk: {mean_salary}")