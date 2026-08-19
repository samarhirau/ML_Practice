for chunk in pd.read_csv("huge_data.csv", chunksize=1000):
    print(chunk.shape)