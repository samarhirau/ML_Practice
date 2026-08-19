from pandas import json_normalize

data = [
    {
        "name": "Samar",
        "age": 22,
        "address": {
            "city": "Pune",
            "state": "Maharashtra"
        }
    }
]

df = json_normalize(data)

print(df)