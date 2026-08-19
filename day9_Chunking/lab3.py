# convert categorical data to numerical and measure memory saving

import pandas as pd


df = pd.read_csv("day4\employees.csv")
print(f"Memory usage before conversion: {df.memory_usage(deep=True).sum()} bytes")

# convert categorical columns to numerical
df["Gender"] = pd.to_numeric(df["Gender"], downcast='integer', errors='coerce')
    
print(f"Memory usage after conversion: {df.memory_usage(deep=True).sum()} bytes")