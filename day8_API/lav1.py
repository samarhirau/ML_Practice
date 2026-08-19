import requests
import pandas as pd

url = "https://api.github.com/users/samarhirau"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print("User:", data.get("login"))
    print("Name:", data.get("name"))
    print("Public Repos:", data.get("public_repos"))

else:
    print("Failed to retrieve data. Status code:", response.status_code)
    
print("Raw JSON data:", data)    
df = pd.json_normalize(data)  # Normalize the JSON data into a flat table
# df = pd.DataFrame([data])    
df.to_json("student_dataset.json", orient="records", lines=True)

print("Data saved to student_dataset.json")
print(df.head())