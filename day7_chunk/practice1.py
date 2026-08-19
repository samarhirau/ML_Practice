from pandas import json_normalize
import pandas as pd

df = pd.read_json("day11\\student_dataset.json", lines=True)

print("nested json before normalization")

df_normalized = json_normalize(df.to_dict(orient="records"))
print(df_normalized.head())


