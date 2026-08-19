for chunk in pd.read_csv("huge_data.csv", chunksize=10000):
    # Process each chunk
    print(chunk.shape)  # Print the shape of the current chunk
    
    
# Convert the "Age" column to numeric type with downcasting to integer  
df["Age"] = pd.to_numeric(df["Age"], downcast="integer")
 
 
# check the memory usage of the DataFrame after downcasting
print(df.memory_usage(deep=True).sum())