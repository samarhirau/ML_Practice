import os 
import pandas as pd
import json

df = pd.read_csv("day4\\student_dataset.csv")
df.to_json("student_dataset.json", orient="records", lines=True)

df1 = pd.read_json("student_dataset.json", lines=True)
df1.to_csv("student_dataset.csv", index=False)

csv_size = os.path.getsize("student_dataset.csv")
json_size = os.path.getsize("student_dataset.json")

print(f"CSV file size: {csv_size} bytes")
print(f"JSON file size: {json_size} bytes")